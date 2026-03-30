import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.evaluation.data_loader import WideSearchDataLoaderHF


def _to_py_scalar(v: Any) -> Any:
    """Convert numpy/pandas scalars into plain Python types for YAML."""
    if v is None:
        return None
    # numpy scalar types
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    # pandas NA/NaN -> None
    try:
        if pd.isna(v):
            return None
    except Exception:
        return None
    return v


def _cell_to_md(v: Any) -> str:
    """Convert a value into a markdown table cell (escape pipes, flatten newlines)."""
    v2 = _to_py_scalar(v)
    if v2 is None:
        return ""
    return (
        str(v2)
        .replace("\r\n", "\n")
        .replace("\n", " ")
        .replace("|", "\\|")
    )


def _df_to_md_table(df) -> str:
    """Convert a DataFrame into a GitHub-flavored markdown table string."""
    df = df.copy().replace({np.nan: None})
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows: list[str] = []
    for _, row in df.iterrows():
        cells = [_cell_to_md(row[col]) for col in columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def dump_all(
    output_dir: str,
    instance_id_whitelist: set[str] | None = None,
    limit: int | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    data_loader = WideSearchDataLoaderHF()
    instance_id_list = data_loader.get_instance_id_list()

    if instance_id_whitelist is not None:
        instance_id_list = [iid for iid in instance_id_list if iid in instance_id_whitelist]

    if limit is not None:
        instance_id_list = instance_id_list[:limit]

    logger.info(f"Dumping {len(instance_id_list)} items to: {output_dir}")

    for idx, instance_id in enumerate(instance_id_list, start=1):
        query = data_loader.load_query_by_instance_id(instance_id)
        # Manual payload to avoid dumping the pandas DataFrame object directly.
        payload = {
            "qid": query.instance_id,
            "query": query.query,
            "evaluation": query.evaluation,
            "language": query.language,
            "answer": _df_to_md_table(query.answer),
        }

        out_path = os.path.join(output_dir, f"{instance_id}.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                payload,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
                width=120,
            )

        if idx % 50 == 0:
            logger.info(f"Dump progress: {idx}/{len(instance_id_list)}")

    logger.info("Dump finished.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/WideSearch",
        help="Output directory for {qid}.yaml",
    )
    parser.add_argument(
        "--instance_ids",
        type=str,
        default="",
        help="Comma-separated instance ids to dump (optional).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of items dumped (0 means no limit).",
    )
    args = parser.parse_args()

    instance_id_whitelist = None
    if args.instance_ids.strip():
        instance_id_whitelist = set([s.strip() for s in args.instance_ids.split(",") if s.strip()])

    limit = args.limit if args.limit > 0 else None

    dump_all(
        output_dir=args.output_dir,
        instance_id_whitelist=instance_id_whitelist,
        limit=limit,
    )


if __name__ == "__main__":
    main()

