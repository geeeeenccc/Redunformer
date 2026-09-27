# WikiText-2 perplexity — SparseGPT 50% on Qwen3-4B — mlsp4

Date: 2026-09-27 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Dense source:** `Qwen/Qwen3-4B`
- **Pruner:** classic SparseGPT, unstructured sparsity **0.5** (WikiText-2 train calibration, block size 128)
- **Pruned model:** `experiments/pruned/qwen3-4b-sparsegpt50`
- **Metric:** WikiText-2 **test** perplexity via lm-eval task `wikitext`
- **Eval:** seed=42, dtype bfloat16, `device_map=auto`, `batch_size=1`, `max_length=2048`
- **Machine:** mlsp4 (`130.83.166.154`)
- **Log:** `experiments/logs/wikitext-ppl-20260927-162850.log`
- **Output:** `experiments/baseline/wikitext/experiments_pruned_qwen3-4b-sparsegpt50_2026-09-27T14-40-01.683145.json`

## SparseGPT 50%

| Metric | Value |
|--------|-------|
| word_perplexity | 26.32 |
| byte_perplexity | 1.843 |
| bits_per_byte | 0.882 |

## Delta (sparsegpt50 − dense)

Dense word_perplexity **18.45** from `reports/weight_level/wikitext_dense_qwen3-4b.md`.

| Metric | Δ |
|--------|---|
| word_perplexity | +7.87 |

At 50% classic SparseGPT, WikiText-2 test perplexity is higher than the dense model and higher than SparseGPT 20% (**18.93**). It stays far below classic magnitude 50% (**105.06**).
