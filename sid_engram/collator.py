import torch

from sid_engram.sid_utils import PAD_SID_ID


class SidEngramDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _target_len(self, lengths):
        max_len = max(lengths)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        return max_len

    def __call__(self, features):
        max_len = self._target_len([len(f["input_ids"]) for f in features])
        max_hist = max([len(f.get("sid_history_ids", [])) for f in features] + [1])
        pad_id = self.tokenizer.pad_token_id

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "sid_history_ids": [],
            "sid_prefix_ids": [],
            "response_anchor_pos": [],
        }
        has_labels = "labels" in features[0]
        if has_labels:
            batch["labels"] = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append([pad_id] * pad_len + f["input_ids"])
            batch["attention_mask"].append([0] * pad_len + f["attention_mask"])
            batch["response_anchor_pos"].append(int(f["response_anchor_pos"]) + pad_len)

            hist = list(f.get("sid_history_ids", []))[:max_hist]
            hist = hist + [PAD_SID_ID] * (max_hist - len(hist))
            batch["sid_history_ids"].append(hist)
            prefixes = [list(x)[:3] + [PAD_SID_ID] * (3 - len(x)) for x in f.get("sid_prefix_ids", [])[:max_hist]]
            prefixes = prefixes + [[PAD_SID_ID, PAD_SID_ID, PAD_SID_ID]] * (max_hist - len(prefixes))
            batch["sid_prefix_ids"].append(prefixes)

            if has_labels:
                batch["labels"].append([-100] * pad_len + f["labels"])

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
