import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LogitProcessor import ConstrainedLogitsProcessor
from sid_engram.cli_utils import run_cli
from sid_engram.dataset import CATEGORY_DICT, SidEngramSFTDataset
from sid_engram.modeling_qwen25_sid_engram import load_sid_engram_checkpoint


def get_hash(x):
    return "-".join(str(_) for _ in x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(
    base_model: str,
    info_file: str,
    category: str,
    test_data_path: str,
    result_json_data: str,
    sid_vocab_dir: str,
    batch_size: int = 4,
    seed: int = 42,
    length_penalty: float = 0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
    if os.path.exists(os.path.join(base_model, "sid_engram.pt")):
        load_sid_engram_checkpoint(model, base_model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    with open(info_file, "r", encoding="utf-8") as f:
        semantic_ids = [line.split("\t")[0].strip() + "\n" for line in f.readlines()]
    info_semantic = [f"### Response:\n{x}" for x in semantic_ids]
    prefix_id = [tokenizer(x).input_ids for x in info_semantic]
    prefix_index = 3
    hash_dict = {}
    for ids in prefix_id:
        ids.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ids)):
            h = get_hash(ids[:i]) if i == prefix_index else get_hash(ids[prefix_index:i])
            hash_dict.setdefault(h, set()).add(ids[i])
    hash_dict = {k: list(v) for k, v in hash_dict.items()}

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(get_hash(input_ids), [])

    eval_category = CATEGORY_DICT.get(category, category)
    val_dataset = SidEngramSFTDataset(
        test_data_path,
        tokenizer,
        sid_vocab_dir=sid_vocab_dir,
        max_len=2560,
        category=category,
        test=True,
    )
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    raw_data = []
    import pandas as pd
    from sid_engram.sid_utils import safe_parse_sid_list

    df = pd.read_csv(test_data_path)
    for _, row in df.iterrows():
        hist = safe_parse_sid_list(row["history_item_sid"])
        raw_data.append({
            "input": f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {', '.join(hist)}",
            "output": str(row["item_sid"]) + "\n",
        })

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate_batch(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, sid_history_ids, response_anchor_pos = [], [], [], []
        sid_prefix_ids = []
        max_hist = max(len(x["sid_history_ids"]) for x in batch)
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * pad_len + x["input_ids"])
            attention_mask.append([0] * pad_len + x["attention_mask"])
            sid_history_ids.append(x["sid_history_ids"] + [-1] * (max_hist - len(x["sid_history_ids"])))
            prefixes = [list(p)[:3] + [-1] * (3 - len(p)) for p in x.get("sid_prefix_ids", [])[:max_hist]]
            prefixes = prefixes + [[-1, -1, -1]] * (max_hist - len(prefixes))
            sid_prefix_ids.append(prefixes)
            response_anchor_pos.append(x["response_anchor_pos"] + pad_len)
        gen_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        clp = ConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams, base_model, model.config.eos_token_id)
        with torch.no_grad():
            output = model.generate(
                torch.tensor(input_ids, device=device),
                attention_mask=torch.tensor(attention_mask, device=device),
                sid_history_ids=torch.tensor(sid_history_ids, device=device),
                sid_prefix_ids=torch.tensor(sid_prefix_ids, device=device),
                response_anchor_pos=torch.tensor(response_anchor_pos, device=device),
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=LogitsProcessorList([clp]),
            )
        completions = output.sequences[:, max_len:]
        text = tokenizer.batch_decode(completions, skip_special_tokens=True)
        text = [x.split("Response:\n")[-1].strip() for x in text]
        return [text[i * num_beams : (i + 1) * num_beams] for i in range(len(text) // num_beams)]

    outputs = []
    for i in range(0, len(encodings), batch_size):
        outputs.extend(evaluate_batch(encodings[i : i + batch_size]))
    for sample, pred in zip(raw_data, outputs):
        sample["predict"] = pred
    os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
    with open(result_json_data, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_cli(main)
