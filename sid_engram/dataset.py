import copy
import os
import random

import pandas as pd
from torch.utils.data import Dataset

from sid_engram.sid_utils import load_json, safe_parse_sid_list, sid_history_to_ids, sid_prefixes


CATEGORY_DICT = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def encode(self, text, bos=False, eos=False):
        ids = self.tokenizer.encode(str(text))
        while ids and self.bos_id is not None and ids[0] == self.bos_id:
            ids = ids[1:]
        while ids and self.eos_id is not None and ids[-1] == self.eos_id:
            ids = ids[:-1]
        if bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids


class SidEngramSFTDataset(Dataset):
    def __init__(
        self,
        train_file,
        tokenizer,
        sid_vocab_dir,
        max_len=512,
        sample=-1,
        seed=42,
        category="Industrial_and_Scientific",
        test=False,
    ):
        self.data = pd.read_csv(train_file)
        if sample > 0 and sample < len(self.data):
            self.data = self.data.sample(sample, random_state=seed).reset_index(drop=True)
        random.seed(seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.test = test
        self.category = CATEGORY_DICT.get(category, category)
        self.sid2idx = load_json(os.path.join(sid_vocab_dir, "sid2idx.json"))
        self.prefix2idx = load_json(os.path.join(sid_vocab_dir, "prefix2idx.json"))
        self.inputs = [self.pre(i) for i in range(len(self.data))]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:
{data_point["output"]}"""

    def get_history(self, row):
        hist = safe_parse_sid_list(row["history_item_sid"])
        history_text = ", ".join(hist)
        target = str(row["item_sid"])
        return {
            "input": f"The user has interacted with items {history_text} in chronological order. Can you predict the next possible item that the user may expect?",
            "output": target + "\n",
            "sid_history": hist,
        }

    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history = self.get_history(self.data.iloc[idx])
        sid_history_ids = sid_history_to_ids(history["sid_history"], self.sid2idx)
        sid_prefix_ids = []
        for sid in history["sid_history"]:
            prefixes = sid_prefixes(sid)
            ids = [int(self.prefix2idx.get(prefix, -1)) for prefix in prefixes[:3]]
            ids = ids + [-1] * (3 - len(ids))
            sid_prefix_ids.append(ids)

        target_item = history["output"]
        prompt_history = copy.deepcopy(history)
        prompt_history["output"] = ""
        prompt = self.generate_prompt(prompt_history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        response_anchor_pos = len(tokens) - 1
        attention_mask = [1] * len(tokens)

        if self.test:
            tokens = tokens[-self.max_len :]
            shift = max(0, len(attention_mask) - self.max_len)
            response_anchor_pos = max(0, response_anchor_pos - shift)
            return {
                "input_ids": tokens,
                "attention_mask": [1] * len(tokens),
                "sid_history_ids": sid_history_ids,
                "sid_prefix_ids": sid_prefix_ids,
                "response_anchor_pos": response_anchor_pos,
                "output": target_item,
            }

        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        shift = max(0, len(tokens) - self.max_len)
        return {
            "input_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
            "labels": labels[-self.max_len :],
            "sid_history_ids": sid_history_ids,
            "sid_prefix_ids": sid_prefix_ids,
            "response_anchor_pos": max(0, response_anchor_pos - shift),
        }
