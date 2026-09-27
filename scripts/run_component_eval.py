#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen3-4B"
PRUNED = Path("experiments/pruned/qwen3-4b-sparsegpt50")
OUTPUT = Path("experiments/baseline/components")
TASKS = ["hellaswag", "piqa", "arc_easy"]
PROJECTIONS = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))\.weight$"
)
VARIANTS = ["dense", "full", "attention", "ffn", *(f"depth_{i}" for i in range(4))]


def selected(name: str, variant: str) -> bool:
    match = PROJECTIONS.fullmatch(name)
    if match is None:
        return False
    layer = int(match.group(1))
    if variant == "attention":
        return match.group(3) is not None
    if variant == "ffn":
        return match.group(4) is not None
    if variant.startswith("depth_"):
        return layer // 9 == int(variant[-1])
    return False


def load_model(pretrained: str | Path) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        pretrained,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )


def replace_weights(model: torch.nn.Module, variant: str) -> dict[str, int]:
    parameters = dict(model.named_parameters())
    names = [name for name in parameters if selected(name, variant)]
    expected = 144 if variant == "attention" else 108 if variant == "ffn" else 63
    if len(names) != expected:
        raise ValueError(f"Expected {expected} projection weights, found {len(names)}")

    count = 0
    zeros = 0
    changed = 0
    with safe_open(PRUNED / "model.safetensors", framework="pt", device="cpu") as source:
        available = set(source.keys())
        missing = sorted(set(names) - available)
        if missing:
            raise ValueError(f"Missing checkpoint weights: {missing[:5]}")
        with torch.no_grad():
            for name in names:
                target = parameters[name]
                weight = source.get_tensor(name)
                if weight.shape != target.shape:
                    raise ValueError(f"Shape mismatch for {name}: {weight.shape} != {target.shape}")
                count += weight.numel()
                zeros += torch.count_nonzero(weight == 0).item()
                converted = weight.to(dtype=target.dtype)
                changed += torch.count_nonzero(target.detach().cpu() != converted).item()
                target.copy_(converted.to(device=target.device))
    return {"tensors": len(names), "weights": count, "zeros": zeros, "changed": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SparseGPT weights by component or depth.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--limit", type=float)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    pretrained = PRUNED if args.variant == "full" else MODEL
    model = load_model(pretrained)
    counts = None
    if args.variant not in {"dense", "full"}:
        counts = replace_weights(model, args.variant)
        print(f"Replaced: {counts}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    evaluator = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto", device="cuda")
    results = lm_eval.simple_evaluate(
        model=evaluator,
        tasks=TASKS,
        batch_size="auto",
        limit=args.limit,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
        log_samples=False,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    path = OUTPUT / f"sparsegpt50_{args.variant}{suffix}.json"
    payload = {"variant": args.variant, "dtype": "float16", "replaced": counts, "evaluation": results}
    path.write_text(json.dumps(payload, indent=2, default=handle_non_serializable), encoding="utf-8")
    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()
