# WikiText-2 perplexity — Magnitude 20% on Qwen3-4B — mlsp4

Date: 2026-09-27 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** `Qwen/Qwen3-4B`
- **Pruner:** classic magnitude, unstructured sparsity **0.2** (no WikiText calibration)
- **Pruned model:** `experiments/pruned/qwen3-4b-magnitude20`
- **Metric:** WikiText-2 **test** perplexity via lm-eval task `wikitext`
- **Eval:** seed=42, dtype bfloat16, `device_map=auto`, `batch_size=1`, `max_length=2048`
- **Machine:** mlsp4 (`130.83.166.154`)
- **Log:** `experiments/logs/wikitext-ppl-20260927-135301.log`
- **Output:** `experiments/baseline/wikitext/experiments_pruned_qwen3-4b-magnitude20_2026-09-27T12-14-59.966583.json`

## Magnitude 20%

| Metric | Value |
|--------|-------|
| word_perplexity | 18.22 |
| byte_perplexity | 1.721 |
| bits_per_byte | 0.783 |

## Delta (magnitude20 − dense)

Dense word_perplexity **18.45** from `reports/weight_level/wikitext_dense_qwen3-4b.md`.

| Metric | Δ |
|--------|---|
| word_perplexity | −0.23 |

At 20% classic magnitude, WikiText-2 test perplexity stays at the dense level (slightly lower here). This matches the downstream-task result that 20% magnitude is nearly free.
