import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sid_engram.collator import SidEngramDataCollator
from sid_engram.cli_utils import run_cli
from sid_engram.dataset import SidEngramSFTDataset
from sid_engram.modeling_qwen25_sid_engram import attach_sid_engram
from sid_engram.sid_engram import SidEngramConfig
from sid_engram.sid_utils import load_json, safe_parse_sid_list


def data(train_file, sid_vocab_dir):
    df = pd.read_csv(train_file)
    sid2idx = load_json(os.path.join(sid_vocab_dir, "sid2idx.json"))
    parsed = df["history_item_sid"].head(100).apply(safe_parse_sid_list)
    print("rows", len(df))
    print("columns", list(df.columns))
    print("num_sids", len(sid2idx))
    print("avg_history_len_head100", parsed.apply(len).mean())
    assert len(df) > 0
    assert len(sid2idx) > 0


def stats(sid_stats_dir):
    trigger2targets = load_json(os.path.join(sid_stats_dir, "trigger2targets.json"))
    summary = load_json(os.path.join(sid_stats_dir, "summary.json"))
    print(summary)
    print("num_triggers", len(trigger2targets))
    assert len(trigger2targets) > 0


def model(base_model, train_file, sid_vocab_dir, batch_size=2):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    attach_sid_engram(model, SidEngramConfig(hidden_size=model.config.hidden_size, layer_ids=[0], memory_dim=64))
    for name, param in model.named_parameters():
        param.requires_grad = "sid_engram" in name
    ds = SidEngramSFTDataset(train_file, tokenizer, sid_vocab_dir, max_len=256, sample=batch_size)
    batch = SidEngramDataCollator(tokenizer)([ds[i] for i in range(batch_size)])
    out = model(**batch)
    print("loss", out.loss.item())
    out.loss.backward()
    grads = [p.grad is not None for n, p in model.named_parameters() if "sid_engram" in n]
    print("sid_engram_grad_params", sum(grads), "/", len(grads))
    assert torch.isfinite(out.loss)
    assert any(grads)


def main(
    mode: str,
    train_file: str = "",
    sid_vocab_dir: str = "",
    sid_stats_dir: str = "",
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    batch_size: int = 2,
):
    if mode == "data":
        data(train_file, sid_vocab_dir)
    elif mode == "stats":
        stats(sid_stats_dir)
    elif mode == "model":
        model(base_model, train_file, sid_vocab_dir, batch_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    run_cli(main)
