import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.evaluation.data_loader import WideSearchQuery, WideSearchResponse
from src.evaluation.metric_utils import (
    metric_function_registry,
    preprocess_function_registry,
)
import src.evaluation.evaluation as evaluation_mod


def stub_llm_judge_column(response, target, criterion, model_config_name="default_eval_config"):
    # 离线 stub：不调用任何外部 LLM，只根据（已在 evaluate 中做完 preprocess 的）字符串相等判断。
    score_list = [1 if str(r) == str(t) else 0 for r, t in zip(response, target)]
    msg_list = [
        "stub llm_judge_column: match" if s == 1 else "stub llm_judge_column: mismatch"
        for s in score_list
    ]
    return score_list, msg_list


def stub_llm_judge_column_metric_scalar(response, target):
    # evaluate_single_query -> metric_call 在 metric_func_name != "llm_judge" 且 != "number_near" 时，
    # 只会调用 metric_func(response, target)，因此这里提供标量版本的 stub。
    s = str(response)
    t = str(target)
    if s == t:
        return 1.0, f"stub llm_judge_column metric: match, response={s}, target={t}"
    return 0.0, f"stub llm_judge_column metric: mismatch, response={s}, target={t}"


def build_query_and_response():
    instance_id = "ws_smoke_all_001"

    required_columns = [
        "id",
        "name",
        "url",
        "phrase",
        "amount",
        "date",
        "judged",
        "judged2",
    ]
    unique_columns = ["id"]

    evaluation = {
        "required": required_columns,
        "unique_columns": unique_columns,
        "eval_pipeline": {
            # exact_match + norm_str
            "name": {
                "preprocess": ["norm_str"],
                "metric": ["exact_match"],
                "criterion": None,
            },
            # url_match + norm_str
            "url": {
                "preprocess": ["norm_str"],
                "metric": ["url_match"],
                "criterion": None,
            },
            # in_match + norm_str
            "phrase": {
                "preprocess": ["norm_str"],
                "metric": ["in_match"],
                "criterion": None,
            },
            # number_near + extract_number
            "amount": {
                "preprocess": ["extract_number"],
                "metric": ["number_near"],
                "criterion": 0.1,
            },
            # date_near + norm_date
            "date": {
                "preprocess": ["norm_date"],
                "metric": ["date_near"],
                "criterion": None,
            },
            # llm_judge 走 llm_judge_column 分支
            "judged": {
                "preprocess": ["norm_str"],
                "metric": ["llm_judge"],
                "criterion": "stub criterion (not used by stub)",
            },
            # 让 metric_function_registry 中的 llm_judge_column 也被 evaluateSingleQuery 调用到
            # （通过 metric 名称 "llm_judge_column" 走 metric_call 分支）。
            "judged2": {
                "preprocess": ["norm_str"],
                "metric": ["llm_judge_column"],
                "criterion": None,
            },
        },
    }

    answer_df = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Alice",
                "url": "https://example.com/a",
                "phrase": "Hello World",
                "amount": "Amount: 100 USD",
                "date": "2026-03-20",
                "judged": "YES",
                "judged2": "YES",
            },
            {
                "id": 2,
                "name": "Bob",
                "url": "https://example.com/x",
                "phrase": "Hello Planet",
                "amount": "Amount: 200 USD",
                "date": "2026-03-10",
                "judged": "NO",
                "judged2": "NO",
            },
        ]
    )

    # evaluate_single_query 内部会用 response.extract_dataframe() 从 ```markdown 表格抽取预测 DataFrame
    # 注意：列名会在解析时 lower + 去空格；因此表头用小写无空格。
    response_text = """```markdown
| id | name | url | phrase | amount | date | judged | judged2 |
|---|---|---|---|---|---|---|---|
| 1 | alice | Visit https://example.com/b | world | Total is 105 USD | 30 Mar 2026 | yes | yes |
| 2 | bob | See https://example.com/x | planet | Total is 210 USD | 2026/03/15 | no | no |
```"""

    query = WideSearchQuery(
        instance_id=instance_id,
        query="dummy query",
        evaluation=evaluation,
        answer=answer_df,
        language="en",
    )
    response = WideSearchResponse(
        instance_id=instance_id,
        response=response_text,
        messages=None,
        trial_idx=None,
    )
    return query, response


def main():
    # 先展示有哪些 preprocess / metric（来自 registry）
    preprocess_names = sorted(list(preprocess_function_registry.keys()))
    metric_names = sorted(list(metric_function_registry.keys()))
    print("preprocess:", preprocess_names)
    print("metrics:", metric_names)

    # 离线 stub，避免 llm_judge 触发外部调用
    evaluation_mod.llm_judge_column = stub_llm_judge_column
    # metric_call 分支下让 llm_judge_column 变为标量可用
    metric_function_registry["llm_judge_column"] = stub_llm_judge_column_metric_scalar

    query, response = build_query_and_response()
    result = evaluation_mod.evaluate_single_query(query=query, response=response)

    print("instance_id:", result.instance_id)
    print("score:", result.score)
    print("precision_by_row:", result.precision_by_row)
    print("recall_by_row:", result.recall_by_row)
    print("f1_by_row:", result.f1_by_row)
    print("precision_by_item:", result.precision_by_item)
    print("recall_by_item:", result.recall_by_item)
    print("f1_by_item:", result.f1_by_item)
    print("msg:\n", result.msg)


if __name__ == "__main__":
    # python scripts/smoke_eval_single_query_all_preprocess_metric.py
    main()

