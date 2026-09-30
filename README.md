# Redunformer — Neuron

Group 5 of the LLM Redundancy Seminar, Summer 2026. We studied redundancy at the level of one feed-forward channel: whether an activation-aware score can find spare neurons, and whether firing rate, depth, model size, duplicates, or a short fine-tune tell the same story.

The seminar report is `reports/group_5/icml/paper.pdf`. The longer results write-up is `reports/group_5/analysis_week15-16.md`. The hypothesis ledger is `reports/group_5/hypothesis_scoreboard.md`.

Masking is a forward hook. Parameter shapes do not change, so the runs make no claim about speed, memory, or FLOPs.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+. Model weights are downloaded on demand and are not in this branch.

```powershell
uv sync
uv run pytest tests/
```

Primary model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), 4-bit on about 8 GB. GPT-2 ran in full precision. Qwen3.5-9B and the Llama-2-7B recovery needed 16 GB.

## Runs

```powershell
uv run python scripts/run_baseline.py --config configs/eval/baseline.yaml
uv run python scripts/run_measurement.py --config configs/measurement/neuron_activations.yaml
uv run python scripts/run_pruning.py --config configs/pruning/neuron_masking.yaml
uv run python scripts/run_pair_ablation.py --config configs/pruning/pair_ablation_qwen3_0.6b.yaml
uv run python scripts/run_merge.py --config configs/pruning/merge_qwen3_0.6b.yaml
uv run python scripts/run_recovery.py --config configs/recovery/lora_qwen3_0.6b.yaml
```

Configs for the other models are under `configs/`. Result JSON is under `experiments/results/`. Activation archives (`.npz`) and model caches are not included.

## Layout

```
configs/          model, measurement, masking, and recovery configs
src/redundancy/   loading, hooks, metrics, masking, recovery
scripts/          command-line entry points
tests/            offline tests
notebooks/group_5/
reports/group_5/  report and results write-up
experiments/results/
```
