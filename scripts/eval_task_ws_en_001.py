import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
import sys
import threading
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.evaluation.data_loader import WideSearchQuery, WideSearchResponse
import src.evaluation.evaluation as evaluation_mod
from src.evaluation import metric_utils as metric_utils_mod


def _extract_markdown_table(text: str) -> str:
    answer_idx = text.lower().find("### answer")
    source = text[answer_idx:] if answer_idx >= 0 else text
    lines = source.splitlines()
    table_lines = [ln.strip() for ln in lines if "|" in ln and ln.strip().startswith("|")]
    if table_lines:
        return "\n".join(table_lines)

    fenced = re.findall(r"```markdown(.*?)```", text, re.DOTALL | re.IGNORECASE)
    for block in fenced:
        candidate = block.strip()
        if candidate and "{data_content}" not in candidate.lower() and "|" in candidate:
            return candidate

    raise ValueError("No markdown table found in input text.")


def _markdown_table_to_df(markdown_table: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in markdown_table.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Markdown table is empty.")

    lines[0] = lines[0].replace(" ", "").lower()
    cleaned = []
    for line in lines:
        if set(line).issubset(set("|- :")):
            continue
        if "|" not in line:
            continue
        cleaned.append("|".join([part.strip() for part in line.split("|")]))
    csv_like = "\n".join(cleaned)
    df = pd.read_csv(StringIO(csv_like), sep="|")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _stub_llm_judge_column(response, target, criterion, model_config_name="default_eval_config"):
    score_list = []
    msg_list = []
    for r, t in zip(response, target):
        rs = _normalize_text(r)
        ts = _normalize_text(t)
        match = rs == ts or rs in ts or ts in rs
        score_list.append(1 if match else 0)
        msg_list.append("stub llm_judge: match" if match else "stub llm_judge: mismatch")
    return score_list, msg_list


def _stub_primary_key_preprocess(values_to_align, reference_values, eval_model_config_name="default_eval_config"):
    return {str(v): str(v) for v in values_to_align}


class SQLiteEvalCache:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    score INTEGER NOT NULL,
                    msg TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS primary_key_cache (
                    cache_key TEXT PRIMARY KEY,
                    mapping_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def get_llm_judge(self, cache_key: str):
        with self.lock:
            cur = self.conn.execute(
                "SELECT score, msg FROM llm_judge_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
        return row if row is not None else None

    def set_llm_judge(self, cache_key: str, score: int, msg: str):
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO llm_judge_cache(cache_key, score, msg, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, int(score), str(msg), time.time()),
            )
            self.conn.commit()

    def get_primary_key_map(self, cache_key: str):
        with self.lock:
            cur = self.conn.execute(
                "SELECT mapping_json FROM primary_key_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def set_primary_key_map(self, cache_key: str, mapping: dict):
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO primary_key_cache(cache_key, mapping_json, created_at)
                VALUES (?, ?, ?)
                """,
                (cache_key, json.dumps(mapping, ensure_ascii=False), time.time()),
            )
            self.conn.commit()


def _stable_hash(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_online_primary_key_preprocess(
    cache: SQLiteEvalCache,
):
    def _online_primary_key_preprocess(values_to_align, reference_values, eval_model_config_name="default_eval_config"):
        cache_key = _stable_hash(
            {
                "kind": "primary_key_preprocess",
                "model": eval_model_config_name,
                "values_to_align": [str(x) for x in values_to_align],
                "reference_values": [str(x) for x in reference_values],
            }
        )
        cached = cache.get_primary_key_map(cache_key)
        if cached is not None:
            return cached
        mapping = metric_utils_mod.primary_key_preprocess(
            values_to_align, reference_values, eval_model_config_name
        )
        cache.set_primary_key_map(cache_key, mapping)
        return mapping

    return _online_primary_key_preprocess


def _build_online_llm_judge_column(
    cache: SQLiteEvalCache,
    max_concurrency: int,
    chunk_size: int,
    llm_timeout: float,
):
    async def _run_chunks(miss_items, criterion: str, model_config_name: str):
        semaphore = asyncio.Semaphore(max_concurrency)
        outputs = {}

        async def worker(chunk):
            idxs = [item[0] for item in chunk]
            responses = [item[1] for item in chunk]
            targets = [item[2] for item in chunk]
            async with semaphore:
                try:
                    coro = asyncio.to_thread(
                        metric_utils_mod.llm_judge_column,
                        responses,
                        targets,
                        criterion,
                        model_config_name,
                    )
                    score_list, msg_list = await asyncio.wait_for(coro, timeout=llm_timeout)
                except Exception as e:
                    score_list = [0] * len(chunk)
                    msg_list = [f"llm judge chunk failed: {e}"] * len(chunk)

            for i, s, m in zip(idxs, score_list, msg_list):
                outputs[i] = (int(s), str(m))

        tasks = []
        for i in range(0, len(miss_items), chunk_size):
            tasks.append(asyncio.create_task(worker(miss_items[i : i + chunk_size])))
        if tasks:
            await asyncio.gather(*tasks)
        return outputs

    def _online_llm_judge_column(response, target, criterion, model_config_name="default_eval_config"):
        score_list = [0] * len(response)
        msg_list = [""] * len(response)
        misses = []

        for idx, (resp, tar) in enumerate(zip(response, target)):
            cache_key = _stable_hash(
                {
                    "kind": "llm_judge",
                    "model": model_config_name,
                    "criterion": str(criterion),
                    "response": str(resp),
                    "target": str(tar),
                }
            )
            cached = cache.get_llm_judge(cache_key)
            if cached is None:
                misses.append((idx, str(resp), str(tar), cache_key))
            else:
                score_list[idx], msg_list[idx] = int(cached[0]), str(cached[1])

        if misses:
            miss_items = [(idx, resp, tar) for idx, resp, tar, _ in misses]
            outputs = asyncio.run(_run_chunks(miss_items, criterion, model_config_name))
            for idx, _, _, cache_key in misses:
                s, m = outputs.get(idx, (0, "llm judge miss fallback"))
                score_list[idx], msg_list[idx] = int(s), str(m)
                cache.set_llm_judge(cache_key, int(s), str(m))

        return score_list, msg_list

    return _online_llm_judge_column


def build_query_from_yaml(yaml_path: Path) -> WideSearchQuery:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    answer_df = _markdown_table_to_df(data["answer"])
    return WideSearchQuery(
        instance_id=data["qid"],
        query=data["query"],
        evaluation=data["evaluation"],
        answer=answer_df,
        language=data.get("language", "en"),
    )


def build_response_from_task(task_md_path: Path, instance_id: str) -> WideSearchResponse:
    task_text = task_md_path.read_text(encoding="utf-8")
    table = _extract_markdown_table(task_text)
    wrapped = f"```markdown\n{table}\n```"
    return WideSearchResponse(instance_id=instance_id, response=wrapped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-md",
        default=str(repo_root / "task.md"),
        help="Path to task markdown that contains your answer table.",
    )
    parser.add_argument(
        "--gold-yaml",
        default=str(repo_root / "data" / "WideSearch" / "ws_en_001.yaml"),
        help="Path to gold yaml file.",
    )
    parser.add_argument(
        "--result-csv",
        default=None,
        help="Optional path to save row/item-level eval detail csv.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use online LLM judge and online primary-key alignment.",
    )
    parser.add_argument(
        "--cache-db",
        default=str(repo_root / ".cache" / "eval_cache.sqlite3"),
        help="SQLite path for online eval cache.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=6,
        help="Max concurrent LLM chunk requests in online mode.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Rows per LLM chunk request in online mode.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=120.0,
        help="Per chunk timeout (seconds) in online mode.",
    )
    args = parser.parse_args()

    if not args.online:
        evaluation_mod.llm_judge_column = _stub_llm_judge_column
        evaluation_mod.primary_key_preprocess = _stub_primary_key_preprocess
    else:
        cache = SQLiteEvalCache(Path(args.cache_db))
        evaluation_mod.llm_judge_column = _build_online_llm_judge_column(
            cache=cache,
            max_concurrency=max(1, args.max_concurrency),
            chunk_size=max(1, args.chunk_size),
            llm_timeout=max(5.0, args.llm_timeout),
        )
        evaluation_mod.primary_key_preprocess = _build_online_primary_key_preprocess(cache=cache)

    query = build_query_from_yaml(Path(args.gold_yaml))
    response = build_response_from_task(Path(args.task_md), query.instance_id)
    result = evaluation_mod.evaluate_single_query(
        query=query,
        response=response,
        result_save_path=args.result_csv,
    )

    print(f"instance_id: {result.instance_id}")
    print(f"score: {result.score}")
    print(f"precision_by_row: {result.precision_by_row}")
    print(f"recall_by_row: {result.recall_by_row}")
    print(f"f1_by_row: {result.f1_by_row}")
    print(f"precision_by_item: {result.precision_by_item}")
    print(f"recall_by_item: {result.recall_by_item}")
    print(f"f1_by_item: {result.f1_by_item}")
    print("msg:")
    print(result.msg)


if __name__ == "__main__":
    # python scripts/eval_task_ws_en_001.py
    main()
