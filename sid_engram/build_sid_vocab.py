import os
import sys
from collections import Counter

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sid_engram.cli_utils import run_cli
from sid_engram.sid_utils import (
    load_json,
    normalize_sid,
    safe_parse_sid_list,
    save_json,
    sid_prefixes,
)


def _collect_csv_sids(path):
    if not path or not os.path.exists(path):
        return [], Counter()
    df = pd.read_csv(path)
    sids = []
    hist_lens = []
    if "item_sid" in df.columns:
        sids.extend(normalize_sid(x) for x in df["item_sid"].dropna().tolist())
    if "history_item_sid" in df.columns:
        for value in df["history_item_sid"].dropna().tolist():
            hist = safe_parse_sid_list(value)
            hist_lens.append(len(hist))
            sids.extend(hist)
    stats = Counter()
    stats["rows"] = len(df)
    stats["avg_history_len_x1000"] = int((sum(hist_lens) / max(1, len(hist_lens))) * 1000)
    return sids, stats


def main(
    index_file: str,
    output_dir: str,
    train_file: str = "",
    valid_file: str = "",
    test_file: str = "",
):
    os.makedirs(output_dir, exist_ok=True)

    all_sids = []
    split_stats = {}
    for split, path in [("train", train_file), ("valid", valid_file), ("test", test_file)]:
        sids, stats = _collect_csv_sids(path)
        all_sids.extend(sids)
        split_stats[split] = dict(stats)

    item2sid = {}
    if index_file and os.path.exists(index_file):
        index = load_json(index_file)
        for item_id, comps in index.items():
            sid = normalize_sid("".join(comps))
            item2sid[str(item_id)] = sid
            all_sids.append(sid)

    all_sids = sorted(set(normalize_sid(sid) for sid in all_sids if normalize_sid(sid)))
    sid2idx = {sid: i for i, sid in enumerate(all_sids)}
    idx2sid = {str(i): sid for sid, i in sid2idx.items()}

    prefixes = sorted({prefix for sid in all_sids for prefix in sid_prefixes(sid)})
    prefix2idx = {prefix: i for i, prefix in enumerate(prefixes)}

    save_json(sid2idx, os.path.join(output_dir, "sid2idx.json"))
    save_json(idx2sid, os.path.join(output_dir, "idx2sid.json"))
    save_json(prefix2idx, os.path.join(output_dir, "prefix2idx.json"))
    save_json(item2sid, os.path.join(output_dir, "item2sid.json"))
    save_json(
        {
            "num_sids": len(sid2idx),
            "num_prefixes": len(prefix2idx),
            "num_items_from_index": len(item2sid),
            "split_stats": split_stats,
            "index_file": index_file,
        },
        os.path.join(output_dir, "summary.json"),
    )
    print(f"Saved SID vocabulary to {output_dir}")
    print(f"num_sids={len(sid2idx)} num_prefixes={len(prefix2idx)}")


if __name__ == "__main__":
    run_cli(main)
