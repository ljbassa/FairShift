#!/usr/bin/env python3
import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate existing FairWire aA eval summary CSVs and draw an LP AUC "
            "vs score-SP Pareto plot."
        )
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--alphaX", type=float, default=0.0)
    parser.add_argument("--x_metric", default="lp/score_sp_abs_gap_mean")
    parser.add_argument("--y_metric", default="lp/auc_mean")
    parser.add_argument("--label_points", choices=["none", "front", "all"], default="all")
    parser.add_argument("--out_prefix", default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def parse_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return text
    if text.lower() == "nan":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def read_one_row(path: Path) -> Dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = [{key: parse_value(value) for key, value in row.items()} for row in csv.DictReader(fp)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def float_tag(value: float) -> str:
    return str(float(value))


def fmt_num(value: Any) -> str:
    if not is_number(value):
        return str(value)
    value = float(value)
    if value.is_integer() and abs(value) < 1e6:
        return str(int(value))
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.4g}"


def summary_pattern(dataset: str, T: int, alpha_x: float) -> re.Pattern:
    return re.compile(
        rf"^{re.escape(dataset)}_(?P<aA>[-+0-9.eE]+)_"
        rf"{re.escape(float_tag(alpha_x))}_cpts__Sync_T{T}__seed_"
        rf"(?P<seed>[0-9]+)\.summary\.csv$"
    )


def load_runs(run_dir: Path, dataset: str, T: int, alpha_x: float) -> List[Dict[str, Any]]:
    eval_dir = run_dir / "eval_outputs"
    pattern = summary_pattern(dataset, T, alpha_x)
    runs: List[Dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*.summary.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        row = read_one_row(path)
        row["aA"] = float(match.group("aA"))
        row["seed"] = int(match.group("seed"))
        row["summary_csv"] = str(path)
        row["model_tag"] = f"{dataset}_{float_tag(row['aA'])}_{float_tag(alpha_x)}_cpts__Sync_T{T}"
        runs.append(row)
    if not runs:
        raise RuntimeError(f"No matching summary CSVs found under {eval_dir}")
    return runs


def aggregate_by_aA(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in runs:
        buckets[float(row["aA"])].append(row)

    metric_keys = sorted(
        {
            key
            for row in runs
            for key, value in row.items()
            if key not in {"aA", "seed"} and is_number(value)
        }
    )

    out: List[Dict[str, Any]] = []
    for alpha_a in sorted(buckets):
        rows = sorted(buckets[alpha_a], key=lambda row: row["seed"])
        agg: Dict[str, Any] = {
            "aA": alpha_a,
            "model_tag": rows[0]["model_tag"],
            "n_runs": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
        }
        for key in metric_keys:
            values = [float(row[key]) for row in rows if is_number(row.get(key))]
            if not values:
                continue
            agg[key] = statistics.fmean(values)
            agg[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        out.append(agg)
    return out


def pareto_front_indices(xs: Sequence[float], ys: Sequence[float]) -> List[int]:
    front: List[int] = []
    for i, (ax, ay) in enumerate(zip(xs, ys)):
        dominated = False
        for j, (bx, by) in enumerate(zip(xs, ys)):
            if i == j:
                continue
            if bx <= ax and by >= ay and (bx < ax or by > ay):
                dominated = True
                break
        if not dominated:
            front.append(i)
    front.sort(key=lambda idx: (xs[idx], -ys[idx]))
    return front


def fieldnames(rows: Sequence[Dict[str, Any]], preferred: Sequence[str]) -> List[str]:
    names: List[str] = []
    seen = set()
    for key in preferred:
        if any(key in row for row in rows):
            names.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                names.append(key)
                seen.add(key)
    return names


def write_csv(rows: Iterable[Dict[str, Any]], path: Path, preferred: Sequence[str]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames(rows, preferred))
        writer.writeheader()
        writer.writerows(rows)


def valid_rows(rows: Sequence[Dict[str, Any]], x_metric: str, y_metric: str) -> List[Dict[str, Any]]:
    valid = [row for row in rows if is_number(row.get(x_metric)) and is_number(row.get(y_metric))]
    if not valid:
        raise RuntimeError(f"No rows contain finite {x_metric} and {y_metric}")
    return valid


def write_compact_metrics(rows: Sequence[Dict[str, Any]], path: Path, x_metric: str, y_metric: str) -> None:
    preferred = [
        "aA",
        "model_tag",
        "n_runs",
        "seeds",
        y_metric,
        f"{y_metric}_std",
        x_metric,
        f"{x_metric}_std",
        "aggregate_lp/auc",
        "aggregate_lp/auc_std",
        "aggregate_lp/score_sp_abs_gap",
        "aggregate_lp/score_sp_abs_gap_std",
        "is_pareto",
    ]
    compact_rows = [{key: row.get(key, "") for key in preferred} for row in rows]
    write_csv(compact_rows, path, preferred)


def plot(rows: Sequence[Dict[str, Any]], args: argparse.Namespace, prefix: str) -> None:
    rows = valid_rows(rows, args.x_metric, args.y_metric)
    xs = [float(row[args.x_metric]) for row in rows]
    ys = [float(row[args.y_metric]) for row in rows]
    xerrs = [float(row.get(f"{args.x_metric}_std", 0.0) or 0.0) for row in rows]
    yerrs = [float(row.get(f"{args.y_metric}_std", 0.0) or 0.0) for row in rows]

    front_idx = pareto_front_indices(xs, ys)
    for idx, row in enumerate(rows):
        row["is_pareto"] = idx in front_idx

    front_rows = [rows[idx] for idx in front_idx]
    run_dir = args.run_dir.resolve()
    png_path = run_dir / f"{prefix}_pareto_curve.png"
    pdf_path = run_dir / f"{prefix}_pareto_curve.pdf"
    front_path = run_dir / f"{prefix}_pareto_front.csv"

    plt.figure(figsize=(8.5, 6.2))
    plt.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt="o", capsize=3, markersize=6, alpha=0.78)

    front_x = [xs[idx] for idx in front_idx]
    front_y = [ys[idx] for idx in front_idx]
    plt.plot(front_x, front_y, linestyle="-", marker="o", linewidth=2)

    if args.label_points in {"front", "all"}:
        annotate = range(len(rows)) if args.label_points == "all" else front_idx
        for idx in annotate:
            row = rows[idx]
            label = f"aA={fmt_num(row['aA'])}"
            if int(row.get("n_runs", 0)) != max(int(item.get("n_runs", 0)) for item in rows):
                label += f"\nn={row.get('n_runs')}"
            plt.annotate(label, (xs[idx], ys[idx]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    title = args.title or f"{args.dataset} FairWire aA Pareto (T={args.T})"
    plt.xlabel(f"{args.x_metric} (lower is better)")
    plt.ylabel(f"{args.y_metric} (higher is better)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.savefig(pdf_path)
    plt.close()

    front_fields = [
        "aA",
        "model_tag",
        "n_runs",
        "seeds",
        args.y_metric,
        f"{args.y_metric}_std",
        args.x_metric,
        f"{args.x_metric}_std",
        "aggregate_lp/auc",
        "aggregate_lp/score_sp_abs_gap",
        "is_pareto",
    ]
    compact_front_rows = [{key: row.get(key, "") for key in front_fields} for row in front_rows]
    write_csv(compact_front_rows, front_path, front_fields)
    print(f"points     : {len(rows)}")
    print(f"front      : {len(front_rows)}")
    print(f"pareto png : {png_path}")
    print(f"pareto pdf : {pdf_path}")
    print(f"front csv  : {front_path}")


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    prefix = args.out_prefix or f"{args.dataset}_aA_T{args.T}"
    runs = load_runs(args.run_dir, args.dataset, args.T, args.alphaX)
    agg_rows = aggregate_by_aA(runs)

    agg_path = args.run_dir / f"{prefix}_summary_agg.csv"
    metrics_path = args.run_dir / f"{prefix}_metrics_auc_sp.csv"
    write_csv(agg_rows, agg_path, ["aA", "model_tag", "n_runs", "seeds", args.y_metric, args.x_metric])
    plot_rows = valid_rows(agg_rows, args.x_metric, args.y_metric)
    plot(plot_rows, args, prefix)
    write_compact_metrics(plot_rows, metrics_path, args.x_metric, args.y_metric)

    print(f"agg csv    : {agg_path}")
    print(f"metric csv : {metrics_path}")


if __name__ == "__main__":
    main()
