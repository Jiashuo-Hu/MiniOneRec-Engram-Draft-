import json
import os
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sid_engram.cli_utils import run_cli
from sid_engram.sid_utils import load_json, safe_parse_sid_list


def main(
    result_json_data: str,
    test_file: str,
    sid_stats_dir: str,
    output_jsonl: str,
    output_md: str = "",
    limit: int = 20,
):
    with open(result_json_data, "r", encoding="utf-8") as f:
        results = json.load(f)
    df = pd.read_csv(test_file)
    trigger2targets = load_json(os.path.join(sid_stats_dir, "trigger2targets.json")) if sid_stats_dir else {}
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    rows = []
    for i, (sample, (_, row)) in enumerate(zip(results, df.iterrows())):
        if i >= limit:
            break
        hist = safe_parse_sid_list(row["history_item_sid"])
        activated = []
        for sid in hist:
            key = f"sid:{sid}"
            if key in trigger2targets:
                activated.append({"trigger": key, "top_targets": trigger2targets[key][:10]})
        rows.append({
            "history_titles": row.get("history_item_title", ""),
            "history_sid": hist,
            "target_title": row.get("item_title", ""),
            "target_sid": row.get("item_sid", ""),
            "predict": sample.get("predict", []),
            "activated_triggers": activated,
        })
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if output_md:
        with open(output_md, "w", encoding="utf-8") as f:
            for idx, row in enumerate(rows, 1):
                f.write(f"## Case {idx}\n\n")
                f.write(f"- Target SID: `{row['target_sid']}`\n")
                f.write(f"- Target title: {row['target_title']}\n")
                f.write(f"- History SID: {', '.join(row['history_sid'])}\n")
                f.write(f"- Predictions: {', '.join(row['predict'][:10])}\n")
                f.write(f"- Activated triggers: {json.dumps(row['activated_triggers'], ensure_ascii=False)}\n\n")
    print(f"Saved {len(rows)} case studies.")


if __name__ == "__main__":
    run_cli(main)
