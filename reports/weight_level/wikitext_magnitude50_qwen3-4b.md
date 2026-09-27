# WikiText-2 perplexity — Magnitude 50% on Qwen3-4B — mlsp4

Date: 2026-09-27 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** `Qwen/Qwen3-4B`
- **Pruner:** classic magnitude, unstructured sparsity **0.5** (no WikiText calibration)
- **Pruned model:** `experiments/pruned/qwen3-4b-magnitude50`
- **Metric:** WikiText-2 **test** perplexity via lm-eval task `wikitext`
- **Eval:** seed=42, dtype bfloat16, `device_map=auto`, `batch_size=1`, `max_length=2048`
- **Machine:** mlsp4 (`130.83.166.154`)
- **Log:** `experiments/logs/wikitext-ppl-20260927-144322.log`
- **Output:** `experiments/baseline/wikitext/experiments_pruned_qwen3-4b-magnitude50_2026-09-27T12-58-39.980799.json`

The first load died during weight loading. The queue retried once and this result is from that second run.

## Magnitude 50%

| Metric | Value |
|--------|-------|
| word_perplexity | 105.06 |
| byte_perplexity | 2.388 |
| bits_per_byte | 1.256 |

## Delta (magnitude50 − dense)

Dense word_perplexity **18.45** from `reports/weight_level/wikitext_dense_qwen3-4b.md`.

| Metric | Δ |
|--------|---|
| word_perplexity | +86.61 |

At 50% classic magnitude, WikiText-2 test perplexity jumps far above the dense model. This matches the downstream-task collapse of global magnitude at this sparsity.
