#!/usr/bin/env python3
"""
Plot baseline comparison and MetaKAN sweep from runs.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_runs(path: Path) -> List[Dict[str, Any]]:
    runs = []
    if not path.exists():
        return runs
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("event") == "run" and rec.get("result"):
            runs.append(rec)
    return runs


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def plot_baseline(runs: List[Dict[str, Any]], out_dir: Path) -> None:
    base = [r for r in runs if r.get("group") == "baseline"]
    if not base:
        return

    labels = [r["model"] for r in base]
    scores = [to_float(r["result"]["test_metric"]) for r in base]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Test accuracy")
    plt.title("MLP vs KAN vs MetaKAN")
    plt.tight_layout()
    out_path = out_dir / "baseline_compare.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hidden_dim_sweep(runs: List[Dict[str, Any]], out_dir: Path) -> None:
    sweep = [r for r in runs if r.get("group") == "metakan_hidden_dim"]
    if not sweep:
        return

    sweep_sorted = sorted(sweep, key=lambda r: r.get("meta", {}).get("hidden_dim", 0))
    x = [r.get("meta", {}).get("hidden_dim", 0) for r in sweep_sorted]
    y = [to_float(r["result"]["test_metric"]) for r in sweep_sorted]

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", color="#E45756")
    plt.xlabel("MetaKAN hidden_dim")
    plt.ylabel("Test accuracy")
    plt.title("MetaKAN hidden_dim sweep")
    plt.tight_layout()
    out_path = out_dir / "metakan_hidden_dim_sweep.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, default=str(Path(__file__).parent / "out" / "runs.jsonl"))
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "out"))
    args = parser.parse_args()

    runs_path = Path(args.runs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_path)

    plot_baseline(runs, out_dir)
    plot_hidden_dim_sweep(runs, out_dir)

    print(f"Plots saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
