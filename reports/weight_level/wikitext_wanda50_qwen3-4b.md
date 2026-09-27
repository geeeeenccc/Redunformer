# WikiText-2 perplexity — Wanda 50% on Qwen3-4B — mlsp4

Date: 2026-09-27 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** `Qwen/Qwen3-4B`
- **Pruner:** Wanda, unstructured sparsity **0.5**
- **Pruned model:** `experiments/pruned/qwen3-4b-wanda50`
- **Metric:** WikiText-2 **test** perplexity via lm-eval task `wikitext`
- **Eval:** seed=42, dtype bfloat16, `device_map=auto`, `batch_size=1`, `max_length=2048`
- **Machine:** mlsp4 (`130.83.166.154`)
- **Log:** `experiments/logs/wikitext-ppl-20260927-161237.log`
- **Output:** `experiments/baseline/wikitext/experiments_pruned_qwen3-4b-wanda50_2026-09-27T14-23-13.638520.json`

Earlier loads on this checkpoint were killed by systemd-oomd during weight loading. This number is from the run that finished after the container was marked so oomd would skip it.

## Wanda 50%

| Metric | Value |
|--------|-------|
| word_perplexity | 23.79 |
| byte_perplexity | 1.809 |
| bits_per_byte | 0.855 |

## Delta (wanda50 − dense)

Dense word_perplexity **18.45** from `reports/weight_level/wikitext_dense_qwen3-4b.md`.

| Metric | Δ |
|--------|---|
| word_perplexity | +5.34 |

At 50% Wanda, WikiText-2 test perplexity is higher than the dense model, and lower than Wanda 50% after DSnoT (**25.77** in `reports/weight_level/wikitext_wanda50-dsnot_qwen3-4b.md`).
