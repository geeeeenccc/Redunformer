# WikiText-2 perplexity — Wanda 50% + DSnoT on Qwen3-4B — mlsp4

Date: 2026-09-27 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** `Qwen/Qwen3-4B`
- **Pruner:** Wanda, unstructured sparsity **0.5**, then DSnoT refine on that mask
- **Pruned model:** `experiments/pruned/qwen3-4b-wanda50-dsnot`
- **Metric:** WikiText-2 **test** perplexity via lm-eval task `wikitext`
- **Eval:** seed=42, dtype bfloat16, `device_map=auto`, `batch_size=1`, `max_length=2048`
- **Machine:** mlsp4 (`130.83.166.154`)
- **Log:** `experiments/logs/wikitext-ppl-20260927-144322.log`
- **Output:** `experiments/baseline/wikitext/experiments_pruned_qwen3-4b-wanda50-dsnot_2026-09-27T13-35-36.880623.json`

## Wanda 50% + DSnoT

| Metric | Value |
|--------|-------|
| word_perplexity | 25.77 |
| byte_perplexity | 1.836 |
| bits_per_byte | 0.877 |

## Delta (wanda50-dsnot − dense)

Dense word_perplexity **18.45** from `reports/weight_level/wikitext_dense_qwen3-4b.md`.

| Metric | Δ |
|--------|---|
| word_perplexity | +7.32 |

At 50% Wanda after DSnoT, WikiText-2 test perplexity is higher than the dense model. The standalone Wanda 50% number is not in yet.
