# Semantic-ID Engram for MiniOneRec

This package adds a Semantic-ID Engram module to MiniOneRec without modifying the original `sft.py` or `evaluate.py`.

The method treats MiniOneRec semantic IDs as memory-addressable recommendation units. Historical SID 1-gram / 2-gram triggers retrieve trainable memory vectors, and those vectors are gated into Qwen2.5 hidden states at the response anchor position.

## Files

```text
sid_utils.py                  SID parsing, trigger keys, deterministic hashing
build_sid_vocab.py            Build full SID and prefix vocabularies
build_sid_engram_stats.py     Build train-only trigger-to-target statistics
dataset.py                    SFT dataset with sid_history_ids and response_anchor_pos
collator.py                   Left-padding collator for token and SID tensors
sid_engram.py                 SidEngram module
modeling_qwen25_sid_engram.py Attach/load/save helpers for Qwen2.5
train_sid_engram_sft.py       SFT entrypoint
evaluate_sid_engram.py        Constrained-decoding evaluation entrypoint
sanity_check.py               Data/stats/model checks
case_study.py                 Qualitative case export
```

## Quick Start

Build vocab:

```bash
cd MiniOneRec
python sid_engram/build_sid_vocab.py \
  --index_file data/Amazon/index/Industrial_and_Scientific.index.json \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --valid_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --test_file data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output_dir output/sid_engram_industrial/vocab
```

Build PMI stats:

```bash
python sid_engram/build_sid_engram_stats.py \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --sid_vocab_dir output/sid_engram_industrial/vocab \
  --output_dir output/sid_engram_industrial/stats \
  --ngram_orders 1,2 \
  --score_type pmi \
  --top_k 50 \
  --recency_lambda 0.2
```

Run checks:

```bash
python sid_engram/sanity_check.py --mode data \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --sid_vocab_dir output/sid_engram_industrial/vocab

python sid_engram/sanity_check.py --mode stats \
  --sid_stats_dir output/sid_engram_industrial/stats
```

Train Engram:

```bash
python sid_engram/train_sid_engram_sft.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --sft_checkpoint output/baseline_qwen25_05b_industrial_sft/final_checkpoint \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --sid_vocab_dir output/sid_engram_industrial/vocab \
  --sid_stats_dir output/sid_engram_industrial/stats \
  --output_dir output/sid_engram_industrial/checkpoints/pmi_1gram_2gram \
  --category Industrial_and_Scientific \
  --engram_init pmi \
  --ngram_orders 1,2 \
  --engram_layer_ids 8,16 \
  --memory_dim 512 \
  --num_hash_buckets 262144 \
  --batch_size 64 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 5e-4 \
  --cutoff_len 512 \
  --freeze_llm true
```

Evaluate:

```bash
python sid_engram/evaluate_sid_engram.py \
  --base_model output/sid_engram_industrial/checkpoints/pmi_1gram_2gram/final_checkpoint \
  --test_data_path data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --info_file data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --category Industrial_and_Scientific \
  --sid_vocab_dir output/sid_engram_industrial/vocab \
  --result_json_data output/sid_engram_industrial/eval/pmi_1gram_2gram.json \
  --batch_size 8 \
  --num_beams 50 \
  --max_new_tokens 256 \
  --length_penalty 0.0

python calc.py \
  --path output/sid_engram_industrial/eval/pmi_1gram_2gram.json \
  --item_path data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

## Recommended Experiments

| Run | Init | Trigger | Trainable |
| --- | --- | --- | --- |
| E1 | random | 1-gram | Engram only |
| E2 | PMI | 1-gram | Engram only |
| E3 | PMI | 1+2-gram | Engram only |

Always compare against the original MiniOneRec SFT baseline and an equal-extra-steps baseline.
