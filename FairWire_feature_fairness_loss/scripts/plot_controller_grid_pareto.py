#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a Pareto curve from a controller-grid summary CSV."
    )
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument(
        "--extra_summary_csv",
        type=Path,
        nargs="*",
        default=[],
        help="Additional existing summary CSVs to include in the same Pareto plot.",
    )
    parser.add_argument(
        "--dedupe_key",
        default="run_name",
        help="CSV column used to deduplicate merged summaries. Use an empty string to disable.",
    )
    parser.add_argument("--out_path", type=Path, default=None)
    parser.add_argument("--front_csv", type=Path, default=None)
    parser.add_argument("--x_metric", default="lp/score_sp_abs_gap_mean")
    parser.add_argument("--y_metric", default="lp/auc_mean")
    parser.add_argument("--xerr_metric", default=None)
    parser.add_argument("--yerr_metric", default=None)
    parser.add_argument("--title", default="Controller LP Pareto: AUC vs score SP gap")
    parser.add_argument("--label_points", choices=["none", "front", "all"], default="front")
    parser.add_argument(
        "--label_fields",
        nargs="+",
        default=["fair_score_k", "fair_score_eta", "fair_weight", "utility_weight"],
    )
    return parser.parse_args()


def parse_csv_value(value: str) -> Any:
    if value is None:
        return value
    text = value.strip()
    if text == "":
        return text
    try:
        return float(text)
    except ValueError:
        return value


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return [{key: parse_csv_value(value) for key, value in row.items()} for row in reader]


def dedupe_rows(rows: List[Dict[str, Any]], dedupe_key: str) -> List[Dict[str, Any]]:
    dedupe_key = (dedupe_key or "").strip()
    if not dedupe_key:
        return rows

    out: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}
    for row in rows:
        key_value = row.get(dedupe_key)
        key_text = "" if key_value is None else str(key_value).strip()
        if not key_text:
            out.append(row)
            continue
        if key_text in index_by_key:
            out[index_by_key[key_text]] = row
        else:
            index_by_key[key_text] = len(out)
            out.append(row)
    return out


def read_merged_summary_rows(
    summary_csv: Path,
    extra_summary_csvs: Sequence[Path],
    dedupe_key: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Load existing summaries first, then the primary summary so freshly
    # regenerated rows win when dedupe_key matches.
    for path in extra_summary_csvs:
        resolved = path.resolve()
        for row in read_csv_rows(resolved):
            row.setdefault("summary_source", str(resolved))
            rows.append(row)

    resolved_summary = summary_csv.resolve()
    for row in read_csv_rows(resolved_summary):
        row.setdefault("summary_source", str(resolved_summary))
        rows.append(row)

    return dedupe_rows(rows, dedupe_key)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def default_out_path(summary_csv: Path, x_metric: str, y_metric: str) -> Path:
    def clean(text: str) -> str:
        return text.replace("/", "_").replace(" ", "_")

    return summary_csv.with_name(f"{summary_csv.stem}_pareto_{clean(y_metric)}_vs_{clean(x_metric)}.jpg")


def std_metric_for(metric: str) -> Optional[str]:
    if metric.endswith("_mean"):
        return f"{metric[:-5]}_std"
    return None


def pareto_front_indices(xs: Sequence[float], ys: Sequence[float]) -> List[int]:
    front = []
    for i, (ax, ay) in enumerate(zip(xs, ys)):
        dominated = False
        for j, (bx, by) in enumerate(zip(xs, ys)):
            if i == j:
                continue
            no_worse = bx <= ax and by >= ay
            strictly_better = bx < ax or by > ay
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    front.sort(key=lambda idx: (xs[idx], -ys[idx]))
    return front


def fmt_num(value: Any) -> str:
    if not is_finite_number(value):
        return str(value)
    value = float(value)
    if value.is_integer() and abs(value) < 1e6:
        return str(int(value))
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.4g}"


def make_label(row: Dict[str, Any], label_fields: Sequence[str]) -> str:
    parts = []
    aliases = {
        "fair_score_k": "k",
        "fair_score_eta": "eta",
        "fair_weight": "fw",
        "utility_weight": "uw",
        "k_tracking_weight": "kw",
    }
    for key in label_fields:
        if key in row and str(row[key]).strip() != "":
            parts.append(f"{aliases.get(key, key)}={fmt_num(row[key])}")
    if parts:
        return ", ".join(parts)
    return str(row.get("run_name", ""))


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_pareto(args) -> Path:
    summary_csv = args.summary_csv.resolve()
    rows = read_merged_summary_rows(summary_csv, args.extra_summary_csv, args.dedupe_key)
    valid = [
        row
        for row in rows
        if is_finite_number(row.get(args.x_metric)) and is_finite_number(row.get(args.y_metric))
    ]
    if not valid:
        raise RuntimeError(
            f"No valid rows with x={args.x_metric!r} and y={args.y_metric!r} in {summary_csv}"
        )

    xs = [float(row[args.x_metric]) for row in valid]
    ys = [float(row[args.y_metric]) for row in valid]
    front_idx = pareto_front_indices(xs, ys)
    front_rows = [valid[i] for i in front_idx]

    xerr_metric = args.xerr_metric if args.xerr_metric is not None else std_metric_for(args.x_metric)
    yerr_metric = args.yerr_metric if args.yerr_metric is not None else std_metric_for(args.y_metric)
    xerrs = [
        float(row.get(xerr_metric, 0.0)) if xerr_metric and is_finite_number(row.get(xerr_metric)) else 0.0
        for row in valid
    ]
    yerrs = [
        float(row.get(yerr_metric, 0.0)) if yerr_metric and is_finite_number(row.get(yerr_metric)) else 0.0
        for row in valid
    ]

    out_path = args.out_path.resolve() if args.out_path is not None else default_out_path(summary_csv, args.x_metric, args.y_metric)
    front_csv = args.front_csv.resolve() if args.front_csv is not None else out_path.with_suffix(".front.csv")

    plt.figure(figsize=(9, 6))
    plt.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt="o", alpha=0.72, capsize=2, markersize=5)

    front_x = np.asarray([xs[i] for i in front_idx], dtype=float)
    front_y = np.asarray([ys[i] for i in front_idx], dtype=float)
    if front_x.size:
        order = np.argsort(front_x)
        plt.plot(front_x[order], front_y[order], linestyle="-", marker="o", linewidth=2)

    if args.label_points in {"front", "all"}:
        annotate_idx = range(len(valid)) if args.label_points == "all" else front_idx
        for idx in annotate_idx:
            plt.annotate(
                make_label(valid[idx], args.label_fields),
                (xs[idx], ys[idx]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    plt.xlabel(f"{args.x_metric} (lower is better)")
    plt.ylabel(f"{args.y_metric} (higher is better)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    write_csv(front_rows, front_csv)

    print(f"summary csv : {summary_csv}")
    if args.extra_summary_csv:
        print("extra csvs  :")
        for path in args.extra_summary_csv:
            print(f"  {path.resolve()}")
    print(f"total rows  : {len(rows)}")
    print(f"valid points: {len(valid)}")
    print(f"front points: {len(front_rows)}")
    print(f"pareto jpg  : {out_path}")
    print(f"front csv   : {front_csv}")
    return out_path


def main():
    args = parse_args()
    plot_pareto(args)


if __name__ == "__main__":
    main()
