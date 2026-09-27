# WikiText-2 perplexity — SparseGPT 20% on Qwen3-4B — mlsp4

Date: 2026-09-27 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** `Qwen/Qwen3-4B`
- **Pruner:** classic SparseGPT, unstructured sparsity **0.2** (WikiText-2 train calibration, block size 128)
- **Pruned model:** `experiments/pruned/qwen3-4b-sparsegpt20`
- **Metric:** WikiText-2 **test** perplexity via lm-eval task `wikitext`
- **Eval:** seed=42, dtype bfloat16, `device_map=auto`, `batch_size=1`, `max_length=2048`
- **Machine:** mlsp4 (`130.83.166.154`)
- **Log:** `experiments/logs/wikitext-ppl-20260927-144322.log`
- **Output:** `experiments/baseline/wikitext/experiments_pruned_qwen3-4b-sparsegpt20_2026-09-27T13-11-30.763112.json`

The first load died during weight loading. The queue retried once and this result is from that second run.

## SparseGPT 20%

| Metric | Value |
|--------|-------|
| word_perplexity | 18.93 |
| byte_perplexity | 1.733 |
| bits_per_byte | 0.793 |

## Delta (sparsegpt20 − dense)

Dense word_perplexity **18.45** from `reports/weight_level/wikitext_dense_qwen3-4b.md`.

| Metric | Δ |
|--------|---|
| word_perplexity | +0.48 |

At 20% classic SparseGPT, WikiText-2 test perplexity stays close to the dense model.
