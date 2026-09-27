# WikiText-2 perplexity — dense Qwen3-4B — mlsp4

Date: 2026-09-27 (mlsp4, RTX 2080 Ti 11 GB)

## Setup

- **Model:** `Qwen/Qwen3-4B` (dense, no pruning)
- **Metric:** WikiText-2 **test** perplexity via lm-eval task `wikitext` (not the training split used for calibration)
- **Eval:** seed=42, dtype bfloat16, `device_map=auto`, `batch_size=1`, `max_length=2048`
- **Machine:** mlsp4 (`130.83.166.154`)
- **Log:** `experiments/logs/wikitext-ppl-20260927-135301.log`
- **Output:** `experiments/baseline/wikitext/Qwen_Qwen3-4B_2026-09-27T12-02-23.223401.json`

Lower perplexity is better. `sample_len=62` is the number of WikiText-2 test documents scored.

## Result

| Metric | Value |
|--------|-------|
| word_perplexity | 18.45 |
| byte_perplexity | 1.725 |
| bits_per_byte | 0.786 |

Use **word_perplexity** as the comparable number against pruned checkpoints.
