# WikiText-2 perplexity — Wanda 20% on Qwen3-4B — mlsp4

Date: 2026-09-27 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** `Qwen/Qwen3-4B`
- **Pruner:** Wanda, unstructured sparsity **0.2**
- **Pruned model:** `experiments/pruned/qwen3-4b-wanda20`
- **Metric:** WikiText-2 **test** perplexity via lm-eval task `wikitext`
- **Eval:** seed=42, dtype bfloat16, `device_map=auto`, `batch_size=1`, `max_length=2048`
- **Machine:** mlsp4 (`130.83.166.154`)
- **Log:** `experiments/logs/wikitext-ppl-20260927-153600.log`
- **Output:** `experiments/baseline/wikitext/experiments_pruned_qwen3-4b-wanda20_2026-09-27T13-50-42.687146.json`

The first load of this retry died during weight loading. The queue retried once and this result is from that second run.

## Wanda 20%

| Metric | Value |
|--------|-------|
| word_perplexity | 18.79 |
| byte_perplexity | 1.731 |
| bits_per_byte | 0.791 |

## Delta (wanda20 − dense)

Dense word_perplexity **18.45** from `reports/weight_level/wikitext_dense_qwen3-4b.md`.

| Metric | Δ |
|--------|---|
| word_perplexity | +0.34 |

At 20% Wanda, WikiText-2 test perplexity stays close to the dense model.
