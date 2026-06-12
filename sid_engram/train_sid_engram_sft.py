import os
import random
import sys

import numpy as np
import torch

# Fix for "Duplicate GPU detected" NCCL error with torchrun
if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

import transformers
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sid_engram.collator import SidEngramDataCollator
from sid_engram.cli_utils import run_cli
from sid_engram.dataset import SidEngramSFTDataset
from sid_engram.modeling_qwen25_sid_engram import (
    attach_sid_engram,
    initialize_sid_engram_from_stats,
    iter_sid_engram_modules,
    save_sid_engram_checkpoint,
)
from sid_engram.sid_engram import SidEngramConfig
from sid_engram.sid_utils import bool_arg, parse_int_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    sft_checkpoint: str = "",
    train_file: str = "",
    eval_file: str = "",
    sid_vocab_dir: str = "",
    sid_stats_dir: str = "",
    output_dir: str = "output/sid_engram",
    category: str = "Industrial_and_Scientific",
    engram_init: str = "random",
    ngram_orders: str = "1",
    engram_layer_ids: str = "8,16",
    memory_dim: int = 512,
    num_hash_buckets: int = 262144,
    batch_size: int = 64,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    cutoff_len: int = 512,
    sample: int = -1,
    seed: int = 42,
    freeze_llm: bool = True,
    train_sid_embeddings: bool = False,
    use_hierarchical_sid: bool = False,
):
    set_seed(seed)
    model_path = sft_checkpoint if sft_checkpoint else base_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    hidden_size = model.config.hidden_size
    config = SidEngramConfig(
        hidden_size=hidden_size,
        memory_dim=memory_dim,
        num_hash_buckets=num_hash_buckets,
        ngram_orders=parse_int_list(ngram_orders),
        use_hierarchical_sid=bool_arg(use_hierarchical_sid),
        layer_ids=parse_int_list(engram_layer_ids),
    )
    attach_sid_engram(model, config)
    if engram_init in {"count", "pmi"} and sid_stats_dir:
        initialize_sid_engram_from_stats(model, tokenizer, sid_stats_dir, sid_vocab_dir, num_hash_buckets)

    if bool_arg(freeze_llm):
        for name, param in model.named_parameters():
            param.requires_grad = "sid_engram" in name
    if bool_arg(train_sid_embeddings):
        model.get_input_embeddings().weight.requires_grad = True

    train_data = SidEngramSFTDataset(train_file, tokenizer, sid_vocab_dir, cutoff_len, sample, seed, category)
    eval_data = SidEngramSFTDataset(eval_file, tokenizer, sid_vocab_dir, cutoff_len, sample, seed, category)

    train_dataset = HFDataset.from_list([train_data[i] for i in range(len(train_data))])
    eval_dataset = HFDataset.from_list([eval_data[i] for i in range(len(eval_data))])
    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)

    args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=0.05,
        save_strategy="steps",
        save_steps=0.05,
        save_total_limit=1,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to=None,
    )
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SidEngramDataCollator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    save_sid_engram_checkpoint(trainer.model, final_dir)
    print("Trainable parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    for module in iter_sid_engram_modules(model):
        print(module.last_stats)


if __name__ == "__main__":
    run_cli(train)
