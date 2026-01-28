#!/usr/bin/env python3
"""
Run MetaKAN image-classification baselines and MetaKAN sweeps.
Defaults to dry-run (prints commands). Use --run to execute.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class RunSpec:
    run_id: str
    group: str
    model: str
    dataset: str
    command: List[str]
    meta: dict


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _parse_result_line(line: str) -> dict:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        return {"raw": line}
    metrics = parts[-5:]
    return {
        "raw": line,
        "timestamp": parts[0],
        "dataset": parts[1],
        "model": parts[2],
        "train_metric": metrics[0],
        "test_metric": metrics[1],
        "num_parameters": metrics[2],
        "total_training_time": metrics[3],
        "avg_time_per_epoch": metrics[4],
    }


def build_runs(args: argparse.Namespace, meta_root: Path) -> List[RunSpec]:
    runs: List[RunSpec] = []
    run_idx = 0
    py = sys.executable

    def next_id(prefix: str) -> str:
        nonlocal run_idx
        run_idx += 1
        return f"{prefix}-{run_idx:03d}"

    # Base comparison: MLP, KAN, MetaKAN
    base_common = [
        "--dataset",
        args.dataset,
        "--layers_width",
        *[str(w) for w in args.layers_width],
        "--batch-size",
        str(args.batch_size),
        "--test-batch-size",
        str(args.test_batch_size),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
    ]

    # MLP
    runs.append(
        RunSpec(
            run_id=next_id("baseline"),
            group="baseline",
            model="MLP",
            dataset=args.dataset,
            command=[py, "train.py", "--model", "MLP", *base_common, "--lr", str(args.lr_mlp)],
            meta={"type": "baseline", "model": "MLP"},
        )
    )

    # KAN
    runs.append(
        RunSpec(
            run_id=next_id("baseline"),
            group="baseline",
            model="KAN",
            dataset=args.dataset,
            command=[
                py,
                "train.py",
                "--model",
                "KAN",
                *base_common,
                "--lr",
                str(args.lr_kan),
                "--grid_size",
                str(args.grid_size),
                "--spline_order",
                str(args.spline_order),
                "--base_activation",
                args.base_activation,
            ],
            meta={"type": "baseline", "model": "KAN"},
        )
    )

    # MetaKAN baseline
    meta_base = [
        "--dataset",
        args.dataset,
        "--layers_width",
        *[str(w) for w in args.layers_width],
        "--batch-size",
        str(args.batch_size),
        "--test-batch-size",
        str(args.test_batch_size),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--optim_set",
        "double",
        "--lr_h",
        str(args.lr_h),
        "--lr_e",
        str(args.lr_e),
        "--embedding_dim",
        str(args.embedding_dim),
        "--hidden_dim",
        str(args.hidden_dim),
        "--grid_size",
        str(args.grid_size),
        "--spline_order",
        str(args.spline_order),
        "--base_activation",
        args.base_activation,
    ]

    runs.append(
        RunSpec(
            run_id=next_id("baseline"),
            group="baseline",
            model="MetaKAN",
            dataset=args.dataset,
            command=[py, "train_meta.py", "--model", "MetaKAN", *meta_base],
            meta={
                "type": "baseline",
                "model": "MetaKAN",
                "hidden_dim": args.hidden_dim,
                "embedding_dim": args.embedding_dim,
            },
        )
    )

    # MetaKAN sweep: hidden_dim
    if args.metakan_hidden_dims:
        for hd in args.metakan_hidden_dims:
            runs.append(
                RunSpec(
                    run_id=next_id("sweep"),
                    group="metakan_hidden_dim",
                    model="MetaKAN",
                    dataset=args.dataset,
                    command=[
                        py,
                        "train_meta.py",
                        "--model",
                        "MetaKAN",
                        *meta_base,
                        "--hidden_dim",
                        str(hd),
                    ],
                    meta={
                        "type": "sweep",
                        "model": "MetaKAN",
                        "hidden_dim": hd,
                        "embedding_dim": args.embedding_dim,
                    },
                )
            )

    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-root", type=str, default=None, help="Path to MetaKAN repo")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--layers-width", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr-mlp", type=float, default=0.01)
    parser.add_argument("--lr-kan", type=float, default=0.01)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--base-activation", type=str, default="silu")
    parser.add_argument("--lr-h", type=float, default=1e-4)
    parser.add_argument("--lr-e", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--metakan-hidden-dims", type=int, nargs="*", default=[8, 16, 32, 64])
    parser.add_argument("--run", action="store_true", help="Execute commands (default is dry-run)")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "out"))
    args = parser.parse_args()

    meta_root = Path(args.meta_root) if args.meta_root else (Path(__file__).parent.parent / "MetaKAN")
    meta_ic = meta_root / "image_classification"
    results_csv = meta_ic / "results" / "results.csv"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"

    runs = build_runs(args, meta_root)

    for run in runs:
        print("\n==>", run.run_id, run.group, run.model)
        print(" ", " ".join(run.command))

        run_record = asdict(run)
        run_record.update({"cwd": str(meta_ic)})

        if not args.run:
            with runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "dry-run", **run_record}) + "\n")
            continue

        before_lines = _read_lines(results_csv)

        proc = subprocess.run(run.command, cwd=str(meta_ic))
        status = proc.returncode

        after_lines = _read_lines(results_csv)
        new_lines = after_lines[len(before_lines) :]

        result = None
        if new_lines:
            result = _parse_result_line(new_lines[-1])

        with runs_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "run",
                        **run_record,
                        "status": status,
                        "result": result,
                    }
                )
                + "\n"
            )

        if status != 0:
            print(f"Run failed with status {status}")
            return status

    print(f"\nWrote run log: {runs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
