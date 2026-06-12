# MiniOneRec 的 Semantic-ID Engram 实验说明

本目录在不改动原始 `sft.py` 和 `evaluate.py` 的前提下，为 MiniOneRec 增加 Semantic-ID Engram。

核心思想：把 MiniOneRec 的语义 ID 作为可寻址记忆单元。用户历史中的 SID 1-gram / 2-gram 触发 Engram memory，经过门控后注入 Qwen2.5 的 hidden states，辅助生成下一个合法 SID。

## 文件说明

```text
sid_utils.py                  SID 解析、trigger key、确定性 hash
build_sid_vocab.py            构建 full SID 与 prefix 词表
build_sid_engram_stats.py     只用 train split 构建 trigger-to-target 统计
dataset.py                    带 sid_history_ids 和 response_anchor_pos 的 SFT 数据集
collator.py                   同时 padding token 与 SID 的 collator
sid_engram.py                 SidEngram 模块
modeling_qwen25_sid_engram.py Qwen2.5 挂载、保存、加载 Engram 的工具
train_sid_engram_sft.py       SFT 训练入口
evaluate_sid_engram.py        约束解码评估入口
sanity_check.py               数据、统计、模型检查
case_study.py                 case study 导出
```

## 1. 构建 SID 词表

```bash
cd MiniOneRec
python sid_engram/build_sid_vocab.py \
  --index_file data/Amazon/index/Industrial_and_Scientific.index.json \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --valid_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --test_file data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output_dir output/sid_engram_industrial/vocab
```

输出：

```text
sid2idx.json
idx2sid.json
prefix2idx.json
item2sid.json
summary.json
```

## 2. 构建 PMI 记忆统计

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

注意：该脚本只能使用 train split，不能使用 valid/test，避免数据泄漏。

## 3. 运行检查

```bash
python sid_engram/sanity_check.py --mode data \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --sid_vocab_dir output/sid_engram_industrial/vocab

python sid_engram/sanity_check.py --mode stats \
  --sid_stats_dir output/sid_engram_industrial/stats
```

如果要检查模型 forward/backward：

```bash
python sid_engram/sanity_check.py --mode model \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --sid_vocab_dir output/sid_engram_industrial/vocab \
  --batch_size 2
```

## 4. 训练 Semantic-ID Engram

建议先训练 MiniOneRec baseline，然后从 baseline checkpoint 加载。

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

默认只训练 Engram，冻结 LLM。这样可以明确验证记忆模块本身是否有效。

## 5. 评估与指标计算

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

重点观察：

```text
HR@10
NDCG@10
CC / invalid generation
coverage
case study 中激活的 trigger 是否合理
```

## 6. 推荐实验矩阵

| 编号 | 初始化 | Trigger | 训练参数 |
| --- | --- | --- | --- |
| E1 | random | 1-gram | Engram only |
| E2 | PMI | 1-gram | Engram only |
| E3 | PMI | 1+2-gram | Engram only |

必须和原始 MiniOneRec SFT baseline、等步数继续训练 baseline 比较。

## 7. 常见问题

- 如果 `CC` 上升：检查 constrained decoding 的 `info_file` 是否和当前 checkpoint 的 SID token 一致。
- 如果 gate 全 0 或全 1：降低学习率，或检查 `response_anchor_pos` 是否被 left padding 正确平移。
- 如果 Engram 参数没有梯度：确认 `remove_unused_columns=False`，并确认 batch 中包含 `sid_history_ids` 与 `response_anchor_pos`。
- 如果 PMI 初始化没有效果：先跑 random 版本和 128 样本 overfit，确认模块链路本身正常。
