import math
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sid_engram.cli_utils import run_cli
from sid_engram.sid_utils import (
    bigram_key,
    load_json,
    normalize_sid,
    prefix_key,
    safe_parse_sid_list,
    save_json,
    sid_key,
    sid_prefixes,
    parse_int_list,
)


def main(
    train_file: str,
    sid_vocab_dir: str,
    output_dir: str,
    ngram_orders: str = "1,2",
    score_type: str = "pmi",
    top_k: int = 50,
    recency_lambda: float = 0.2,
    use_hierarchical_sid: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    sid2idx = load_json(os.path.join(sid_vocab_dir, "sid2idx.json"))
    orders = set(parse_int_list(ngram_orders))
    df = pd.read_csv(train_file)

    pair_counts = defaultdict(Counter)
    key_counts = Counter()
    target_counts = Counter()
    total_weight = 0.0

    for _, row in df.iterrows():
        hist = safe_parse_sid_list(row["history_item_sid"])
        target = normalize_sid(row["item_sid"])
        if target not in sid2idx:
            continue
        target_counts[target] += 1.0
        for pos, sid in enumerate(hist):
            distance = len(hist) - pos
            weight = math.exp(-recency_lambda * max(0, distance - 1))
            keys = []
            if 1 in orders:
                keys.append(sid_key(sid))
            if use_hierarchical_sid:
                keys.extend(prefix_key(prefix) for prefix in sid_prefixes(sid))
            for key in keys:
                pair_counts[key][target] += weight
                key_counts[key] += weight
                total_weight += weight
        if 2 in orders and len(hist) >= 2:
            for i in range(1, len(hist)):
                distance = len(hist) - i
                weight = math.exp(-recency_lambda * max(0, distance - 1))
                key = bigram_key(hist[i - 1], hist[i])
                pair_counts[key][target] += weight
                key_counts[key] += weight
                total_weight += weight

    target_total = sum(target_counts.values()) or 1.0
    trigger2targets = {}
    trigger2scores = {}
    trigger2count = {}

    for key, targets in pair_counts.items():
        scored = []
        key_total = key_counts[key] or 1.0
        for target, count in targets.items():
            if score_type == "count":
                score = count
            else:
                p_t_given_k = count / key_total
                p_t = target_counts[target] / target_total
                pmi = math.log((p_t_given_k + 1e-12) / (p_t + 1e-12))
                score = math.sqrt(count) * max(pmi, 0.0)
                if score_type == "pmi_recency":
                    score *= min(1.0, key_total / 5.0)
            if score > 0:
                scored.append((target, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]
        if scored:
            trigger2targets[key] = [target for target, _ in scored]
            trigger2scores[key] = [score for _, score in scored]
            trigger2count[key] = key_total

    save_json(trigger2targets, os.path.join(output_dir, "trigger2targets.json"))
    save_json(trigger2scores, os.path.join(output_dir, "trigger2scores.json"))
    save_json(trigger2count, os.path.join(output_dir, "trigger2count.json"))
    save_json(dict(target_counts), os.path.join(output_dir, "target_popularity.json"))
    save_json(
        {
            "train_file": train_file,
            "score_type": score_type,
            "ngram_orders": sorted(orders),
            "top_k": top_k,
            "recency_lambda": recency_lambda,
            "use_hierarchical_sid": use_hierarchical_sid,
            "num_triggers": len(trigger2targets),
            "num_targets": len(target_counts),
            "total_weight": total_weight,
        },
        os.path.join(output_dir, "summary.json"),
    )
    print(f"Saved SID Engram stats to {output_dir}")
    print(f"num_triggers={len(trigger2targets)} num_targets={len(target_counts)}")


if __name__ == "__main__":
    run_cli(main)
