#!/usr/bin/env python3
"""Generate summary tables (Markdown/CSV) from run logs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_table(rows: List[Dict[str, Any]], headers: List[str], out_md: Path, out_csv: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in headers})

    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        md += "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n"
    out_md.write_text(md)


def _read_num_parameters_from_xlsx(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        df = pd.read_excel(path)
        last = df.iloc[-1]
        value = last.get("num_parameters")
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def summarize_image(out_dir: Path) -> None:
    runs = load_jsonl(out_dir / "runs.jsonl")
    rows = []
    for r in runs:
        if r.get("event") != "run" or not r.get("result"):
            continue
        res = r["result"]
        rows.append(
            {
                "dataset": r.get("dataset"),
                "model": r.get("model"),
                "group": r.get("group"),
                "test_metric": res.get("test_metric"),
                "num_parameters": res.get("num_parameters"),
                "total_time_s": res.get("total_training_time"),
                "gpu_peak_mb": res.get("gpu_peak_mb"),
            }
        )

    headers = ["dataset", "model", "group", "test_metric", "num_parameters", "total_time_s", "gpu_peak_mb"]
    write_table(rows, headers, out_dir / "image_summary.md", out_dir / "image_summary.csv")


def summarize_ff(out_dir: Path) -> None:
    runs = load_jsonl(out_dir / "runs_function_fitting.jsonl")
    rows = []
    for r in runs:
        if r.get("event") != "run" or not r.get("result"):
            continue
        res = r["result"]
        rows.append(
            {
                "dataset": r.get("dataset"),
                "model": r.get("model"),
                "group": r.get("group"),
                "test_metric": res.get("test_metric"),
                "num_parameters": res.get("num_parameters"),
                "total_time_s": res.get("total_training_time"),
                "gpu_peak_mb": res.get("gpu_peak_mb"),
            }
        )

    headers = ["dataset", "model", "group", "test_metric", "num_parameters", "total_time_s", "gpu_peak_mb"]
    write_table(rows, headers, out_dir / "ff_summary.md", out_dir / "ff_summary.csv")


def summarize_pde(out_dir: Path, dataset: str) -> None:
    runs = load_jsonl(out_dir / f"runs_pde_{dataset}.jsonl")
    rows = []
    for r in runs:
        if r.get("event") != "run" or not r.get("result"):
            continue
        res = r["result"]
        num_parameters = res.get("num_parameters")
        if num_parameters is None and res.get("file"):
            num_parameters = _read_num_parameters_from_xlsx(Path(res["file"]))
        rows.append(
            {
                "dataset": r.get("dataset"),
                "model": r.get("model"),
                "group": r.get("group"),
                "L2": res.get("L2"),
                "L1": res.get("L1"),
                "num_parameters": num_parameters,
                "gpu_peak_mb": res.get("gpu_peak_mb"),
            }
        )

    headers = ["dataset", "model", "group", "L2", "L1", "num_parameters", "gpu_peak_mb"]
    write_table(rows, headers, out_dir / f"pde_{dataset}_summary.md", out_dir / f"pde_{dataset}_summary.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "out"))
    parser.add_argument("--pde-dataset", type=str, default="Poisson")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summarize_image(out_dir)
    summarize_ff(out_dir)
    summarize_pde(out_dir, args.pde_dataset)
    print(f"Summaries saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
