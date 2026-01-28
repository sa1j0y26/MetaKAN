#!/usr/bin/env python3
"""
Run MetaKAN PDE experiments (Poisson/Allen-Cahn) and capture L2 metrics.
Defaults to dry-run (prints commands). Use --run to execute.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class RunSpec:
    run_id: str
    group: str
    model: str
    dataset: str
    command: List[str]
    meta: dict


def _latest_xlsx(saved_dir: Path) -> Optional[Path]:
    if not saved_dir.exists():
        return None
    files = sorted(saved_dir.glob("*.xlsx"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _read_l2(path: Path) -> dict:
    df = pd.read_excel(path)
    last = df.iloc[-1]
    return {
        "file": str(path),
        "loss": float(last.get("loss", float("nan"))),
        "L2": float(last.get("L2", float("nan"))),
        "L1": float(last.get("L1", float("nan"))),
    }


def build_runs(args: argparse.Namespace, meta_root: Path) -> List[RunSpec]:
    runs: List[RunSpec] = []
    run_idx = 0
    py = sys.executable

    def next_id(prefix: str) -> str:
        nonlocal run_idx
        run_idx += 1
        return f"{prefix}-{run_idx:03d}"

    base = [
        "--dim",
        str(args.dim),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--lr_h",
        str(args.lr_h),
        "--lr_e",
        str(args.lr_e),
        "--PINN_h",
        str(args.pinn_h),
        "--PINN_L",
        str(args.pinn_l),
        "--N_f",
        str(args.n_f),
        "--N_test",
        str(args.n_test),
        "--batch_size",
        str(args.batch_size),
        "--grid",
        str(args.grid),
        "--k",
        str(args.k),
        "--embedding_dim",
        str(args.embedding_dim),
        "--hidden_dim",
        str(args.hidden_dim),
        "--SEED",
        str(args.seed),
    ]

    if args.dataset == "Poisson":
        script = "Poisson.py"
        dataset_arg = "Poisson"
    else:
        script = "AllenCahn.py"
        dataset_arg = "Allen_Cahn"

    for model in ["MLP", "KAN", "MetaKAN"]:
        runs.append(
            RunSpec(
                run_id=next_id("baseline"),
                group="baseline",
                model=model,
                dataset=args.dataset,
                command=[
                    py,
                    script,
                    "--dataset",
                    dataset_arg,
                    "--model",
                    model,
                    *base,
                ],
                meta={"type": "baseline", "model": model},
            )
        )

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
                        script,
                        "--dataset",
                        dataset_arg,
                        "--model",
                        "MetaKAN",
                        *base,
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
    parser.add_argument("--dataset", type=str, default="Poisson", choices=["Poisson", "Allen_Cahn"])
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-h", type=float, default=1e-3)
    parser.add_argument("--lr-e", type=float, default=1e-2)
    parser.add_argument("--pinn-h", type=int, default=128)
    parser.add_argument("--pinn-l", type=int, default=4)
    parser.add_argument("--n-f", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--metakan-hidden-dims", type=int, nargs="*", default=[8, 16, 32, 64])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", action="store_true", help="Execute commands (default is dry-run)")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "out"))
    args = parser.parse_args()

    meta_root = Path(args.meta_root) if args.meta_root else Path(__file__).parent.parent
    meta_pde = meta_root / "solving_pde"
    saved_dir = meta_pde / "saved_loss_l2"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / f"runs_pde_{args.dataset}.jsonl"

    runs = build_runs(args, meta_root)

    for run in runs:
        print("\n==>", run.run_id, run.group, run.model)
        print(" ", " ".join(run.command))

        run_record = asdict(run)
        run_record.update({"cwd": str(meta_pde)})

        if not args.run:
            with runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "dry-run", **run_record}) + "\n")
            continue

        before_latest = _latest_xlsx(saved_dir)
        before_mtime = before_latest.stat().st_mtime if before_latest else None

        proc = subprocess.run(run.command, cwd=str(meta_pde))
        status = proc.returncode

        after_latest = _latest_xlsx(saved_dir)
        result = None
        if after_latest:
            after_mtime = after_latest.stat().st_mtime
            if before_mtime is None or after_mtime >= before_mtime:
                result = _read_l2(after_latest)

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
