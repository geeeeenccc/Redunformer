#!/usr/bin/env python3
"""Regenerate cosine-similarity heatmaps from saved measurement JSONs (no model rerun needed).

Usage:
    python scripts/replot_heatmaps.py experiments/measurement_gpt2-medium_*.json
    python scripts/replot_heatmaps.py experiments/measurement_*.json --figures-dir reports/group9/figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redundancy.plotting import plot_similarity_heatmap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_files", nargs="+", help="measurement_*.json files containing 'cosine_similarity_matrix'.")
    p.add_argument("--figures-dir", default="reports/group9/figures")
    p.add_argument("--title", default="Hidden-state cosine similarity (centered)")
    args = p.parse_args()

    for json_file in args.json_files:
        payload = json.loads(Path(json_file).read_text(encoding="utf-8"))
        matrix = np.asarray(payload["cosine_similarity_matrix"], dtype=float)
        model_name = payload.get("config", {}).get("model_name") or payload["model_name"]
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        print(f"{json_file}: {model_name}, {matrix.shape[0] - 1} blocks")
        plot_similarity_heatmap(matrix, f"{args.figures_dir}/cosine_heatmap_{safe_name}.png", title=args.title)


if __name__ == "__main__":
    main()
