#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NON_METRIC_KEYS = {
    "model_tag",
    "model_path",
    "seed",
    "sample_returncode",
    "eval_returncode",
    "raw_log",
    "sample_cmd",
    "eval_cmd",
    "pt_path",
    "per_graph_csv",
    "summary_csv",
    "sample_error",
    "eval_error",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run FairWire sample.py -> evaluate_generated_graphs.py over multiple checkpoints "
            "and seeds, aggregate the results, and draw a Pareto curve."
        )
    )
    p.add_argument("--repo_dir", type=str, default=".", help="Path to FairWire repo root")
    p.add_argument("--python_exec", type=str, required=True, help="Python executable for FairWire scripts")
    p.add_argument("--model_paths", type=str, nargs="*", default=[], help="Explicit checkpoint paths")
    p.add_argument("--model_globs", type=str, nargs="*", default=[], help="Glob patterns relative to repo_dir for checkpoint discovery")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name for evaluate_generated_graphs.py")
    p.add_argument("--num_samples", type=int, default=64, help="Number of generated graphs per run")
    p.add_argument("--sample_gpu", type=int, default=0, help="GPU id forwarded to sample.py")
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="Sampling/eval seeds")
    p.add_argument("--out_dir", type=str, default="fair_grid_results", help="Directory for outputs")
    p.add_argument("--fail_fast", action="store_true", help="Stop on first subprocess failure")
    p.add_argument(
        "--plot_x_metric",
        type=str,
        default="lp/score_sp_abs_gap_mean",
        help="Per-run summary metric to minimize on the x-axis",
    )
    p.add_argument(
        "--plot_y_metric",
        type=str,
        default="lp/auc_mean",
        help="Per-run summary metric to maximize on the y-axis",
    )
    p.add_argument("--label_points", choices=["none", "front", "all"], default="front")
    p.add_argument("--plot_title", type=str, default="Pareto curve: fairness gap vs AUC")
    p.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments forwarded to evaluate_generated_graphs.py. Put them after '--'. "
            "Do not include --graph_path, --dataset, --seed, --out_per_graph_csv, or --out_summary_csv."
        ),
    )
    args = p.parse_args()
    if args.eval_args and args.eval_args[0] == "--":
        args.eval_args = args.eval_args[1:]
    return args


def sanitize_tag(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def strip_flag_with_value(argv: List[str], flag: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def prepare_eval_args(argv: List[str]) -> List[str]:
    cleaned = list(argv)
    if cleaned and cleaned[0].endswith("evaluate_generated_graphs.py"):
        cleaned = cleaned[1:]
    for flag in ("--graph_path", "--dataset", "--seed", "--out_per_graph_csv", "--out_summary_csv"):
        cleaned = strip_flag_with_value(cleaned, flag)
    return cleaned


def collect_model_paths(repo_dir: Path, model_paths: Sequence[str], model_globs: Sequence[str]) -> List[Path]:
    discovered = set()
    for model_path in model_paths:
        discovered.add(str((repo_dir / model_path).resolve() if not Path(model_path).is_absolute() else Path(model_path).resolve()))
    for pattern in model_globs:
        for path in repo_dir.glob(pattern):
            if path.is_file():
                discovered.add(str(path.resolve()))
    paths = [Path(p) for p in sorted(discovered)]
    if not paths:
        raise ValueError("No checkpoint paths were provided or discovered.")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))
    return paths


def try_parse_value(value: str):
    if value == "":
        return value
    lower = value.lower()
    if lower == "nan":
        return float("nan")
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_summary_csv(path: Path) -> Dict:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one summary row in {path}, got {len(rows)}")
    return {k: try_parse_value(v) for k, v in rows[0].items()}


def fmt_num(x):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if abs(x) >= 1e4 or (0 < abs(x) < 1e-4):
            return f"{x:.6e}"
        return f"{x:.6f}"
    return str(x)


def ordered_fieldnames(rows: List[Dict]) -> List[str]:
    preferred = [
        "model_tag",
        "model_path",
        "seed",
        "n_runs",
        "n_success",
        "seeds",
        "sample_returncode",
        "eval_returncode",
        "lp/auc_mean",
        "lp/sp_abs_gap_mean",
        "fair_abs_gap_mean",
        "value/fair_abs_gap_mean",
        "value/linkpred_auc_mean",
        "lp/score_sp_abs_gap_mean",
        "overlap/auc_mean",
        "lp/auc_mean_seed_mean",
        "lp/auc_mean_seed_std",
        "lp/sp_abs_gap_mean_seed_mean",
        "lp/sp_abs_gap_mean_seed_std",
        "fair_abs_gap_mean_seed_mean",
        "fair_abs_gap_mean_seed_std",
        "value/fair_abs_gap_mean_seed_mean",
        "value/fair_abs_gap_mean_seed_std",
        "value/linkpred_auc_mean_seed_mean",
        "value/linkpred_auc_mean_seed_std",
        "lp/score_sp_abs_gap_mean_seed_mean",
        "lp/score_sp_abs_gap_mean_seed_std",
        "overlap/auc_mean_seed_mean",
        "overlap/auc_mean_seed_std",
        "raw_log",
        "sample_cmd",
        "eval_cmd",
        "pt_path",
        "per_graph_csv",
        "summary_csv",
        "sample_error",
        "eval_error",
    ]
    out: List[str] = []
    seen = set()
    for key in preferred:
        if any(key in row for row in rows):
            out.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                out.append(key)
                seen.add(key)
    return out


def write_csv(rows: List[Dict], path: Path) -> None:
    keys = ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(float(x))


def numeric_metric_keys(rows: List[Dict]) -> List[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if key in NON_METRIC_KEYS:
                continue
            if is_number(value):
                keys.add(key)
    return sorted(keys)


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["model_tag"], []).append(row)

    metric_keys = numeric_metric_keys([r for r in rows if r.get("sample_returncode") == 0 and r.get("eval_returncode") == 0])
    agg_rows: List[Dict] = []
    for model_tag, bucket in sorted(grouped.items(), key=lambda x: x[0]):
        success_rows = [r for r in bucket if r.get("sample_returncode") == 0 and r.get("eval_returncode") == 0]
        agg = {
            "model_tag": model_tag,
            "model_path": bucket[0]["model_path"],
            "n_runs": len(bucket),
            "n_success": len(success_rows),
            "seeds": ",".join(str(r["seed"]) for r in sorted(bucket, key=lambda r: r["seed"])),
        }
        if not success_rows:
            agg_rows.append(agg)
            continue
        for key in metric_keys:
            vals = [float(r[key]) for r in success_rows if key in r and is_number(r[key])]
            if not vals:
                continue
            agg[f"{key}_seed_mean"] = statistics.fmean(vals)
            agg[f"{key}_seed_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
    return agg_rows


def pareto_front_indices(xs: Sequence[float], ys: Sequence[float]) -> List[int]:
    idxs = list(range(len(xs)))
    front: List[int] = []
    for i in idxs:
        dominated = False
        for j in idxs:
            if i == j:
                continue
            no_worse_x = xs[j] <= xs[i]
            no_worse_y = ys[j] >= ys[i]
            strictly_better = xs[j] < xs[i] or ys[j] > ys[i]
            if no_worse_x and no_worse_y and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    front.sort(key=lambda i: (xs[i], -ys[i]))
    return front


def plot_pareto(agg_rows: List[Dict], out_dir: Path, x_metric: str, y_metric: str, title: str, label_points: str):
    x_mean_key = f"{x_metric}_seed_mean"
    x_std_key = f"{x_metric}_seed_std"
    y_mean_key = f"{y_metric}_seed_mean"
    y_std_key = f"{y_metric}_seed_std"

    valid = [
        row for row in agg_rows
        if is_number(row.get(x_mean_key)) and is_number(row.get(y_mean_key)) and row.get("n_success", 0) > 0
    ]
    if not valid:
        raise RuntimeError(
            f"No aggregated rows contain {x_mean_key} and {y_mean_key}. "
            "Check that sample.py and evaluate_generated_graphs.py completed successfully."
        )

    xs = [float(row[x_mean_key]) for row in valid]
    ys = [float(row[y_mean_key]) for row in valid]
    xerrs = [float(row.get(x_std_key, 0.0) or 0.0) for row in valid]
    yerrs = [float(row.get(y_std_key, 0.0) or 0.0) for row in valid]
    labels = [row["model_tag"] for row in valid]

    front_idx = pareto_front_indices(xs, ys)
    front_rows = [valid[i] for i in front_idx]

    plt.figure(figsize=(8, 6))
    plt.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt="o", capsize=3)

    front_x = [xs[i] for i in front_idx]
    front_y = [ys[i] for i in front_idx]
    plt.plot(front_x, front_y, linestyle="-", marker="o")

    if label_points in {"front", "all"}:
        annotate_idx: Iterable[int] = range(len(valid)) if label_points == "all" else front_idx
        for i in annotate_idx:
            plt.annotate(labels[i], (xs[i], ys[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    plt.xlabel(f"{x_metric} (lower is better)")
    plt.ylabel(f"{y_metric} (higher is better)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = out_dir / "pareto_curve.png"
    pdf_path = out_dir / "pareto_curve.pdf"
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    pareto_csv = out_dir / "pareto_front.csv"
    write_csv(front_rows, pareto_csv)
    return png_path, pdf_path, pareto_csv


def build_model_tag(model_path: Path) -> str:
    return sanitize_tag(f"{model_path.parent.name}__{model_path.stem}")


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"repo_dir does not exist: {repo_dir}")
    if not (repo_dir / "sample.py").exists():
        raise FileNotFoundError(f"sample.py not found in repo_dir: {repo_dir}")
    if not (repo_dir / "evaluate_generated_graphs.py").exists():
        raise FileNotFoundError(f"evaluate_generated_graphs.py not found in repo_dir: {repo_dir}")

    model_paths = collect_model_paths(repo_dir, args.model_paths, args.model_globs)
    eval_args = prepare_eval_args(args.eval_args)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = out_dir / "generated_graphs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "eval_outputs"
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    total_runs = len(model_paths) * len(args.seeds)
    run_idx = 0

    print(f"Checkpoints : {len(model_paths)}")
    print(f"Seeds       : {args.seeds}")
    print(f"Total runs  : {total_runs}")
    print("Eval args   : " + " ".join(eval_args))

    for model_path in model_paths:
        model_tag = build_model_tag(model_path)
        for seed in args.seeds:
            run_idx += 1
            run_name = f"{model_tag}__seed_{seed}"
            raw_path = raw_dir / f"{run_name}.txt"
            pt_path = generated_dir / f"{run_name}.pyg.pt"
            per_graph_csv = eval_dir / f"{run_name}.per_graph.csv"
            summary_csv = eval_dir / f"{run_name}.summary.csv"

            sample_cmd = [
                args.python_exec,
                "sample.py",
                "--model_path", str(model_path),
                "--num_samples", str(args.num_samples),
                "--gpu", str(args.sample_gpu),
                "--seed", str(seed),
                "--save_pt_path", str(pt_path),
                "--skip_internal_eval",
            ]
            eval_cmd = [
                args.python_exec,
                "evaluate_generated_graphs.py",
                "--graph_path", str(pt_path),
                "--dataset", args.dataset,
                "--seed", str(seed),
                "--out_per_graph_csv", str(per_graph_csv),
                "--out_summary_csv", str(summary_csv),
                *eval_args,
            ]

            print(f"[{run_idx}/{total_runs}] {run_name}")
            sample_proc = subprocess.run(
                sample_cmd,
                cwd=str(repo_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            row = {
                "model_tag": model_tag,
                "model_path": str(model_path),
                "seed": seed,
                "sample_returncode": sample_proc.returncode,
                "pt_path": str(pt_path),
                "per_graph_csv": str(per_graph_csv),
                "summary_csv": str(summary_csv),
                "sample_cmd": " ".join(sample_cmd),
                "eval_cmd": " ".join(eval_cmd),
                "raw_log": str(raw_path),
            }

            eval_proc = None
            if sample_proc.returncode == 0:
                eval_proc = subprocess.run(
                    eval_cmd,
                    cwd=str(repo_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                row["eval_returncode"] = eval_proc.returncode
                if eval_proc.returncode == 0:
                    try:
                        row.update(parse_summary_csv(summary_csv))
                    except Exception as e:
                        row["eval_error"] = f"Failed to parse summary CSV: {e}"
                else:
                    row["eval_error"] = "evaluate_generated_graphs.py failed; see raw_log"
            else:
                row["eval_returncode"] = -1
                row["sample_error"] = "sample.py failed; see raw_log"

            with raw_path.open("w", encoding="utf-8") as f:
                f.write("SAMPLE CMD:\n")
                f.write(" ".join(sample_cmd) + "\n\n")
                f.write("SAMPLE STDOUT:\n")
                f.write(sample_proc.stdout)
                f.write("\n\nSAMPLE STDERR:\n")
                f.write(sample_proc.stderr)
                f.write(f"\n\nSAMPLE RETURNCODE: {sample_proc.returncode}\n")
                f.write("\n\nEVAL CMD:\n")
                f.write(" ".join(eval_cmd) + "\n\n")
                if eval_proc is not None:
                    f.write("EVAL STDOUT:\n")
                    f.write(eval_proc.stdout)
                    f.write("\n\nEVAL STDERR:\n")
                    f.write(eval_proc.stderr)
                    f.write(f"\n\nEVAL RETURNCODE: {eval_proc.returncode}\n")

            rows.append(row)
            if args.fail_fast and (sample_proc.returncode != 0 or row.get("eval_returncode") != 0):
                break
        else:
            continue
        break

    rows_sorted = sorted(rows, key=lambda r: (r["model_tag"], r["seed"]))
    agg_rows = aggregate_rows(rows_sorted)

    with (out_dir / "summary_long.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    with (out_dir / "summary_agg.json").open("w", encoding="utf-8") as f:
        json.dump(agg_rows, f, ensure_ascii=False, indent=2)

    write_csv(rows_sorted, out_dir / "summary_long.csv")
    write_csv(agg_rows, out_dir / "summary_agg.csv")

    png_path, pdf_path, pareto_csv = plot_pareto(
        agg_rows=agg_rows,
        out_dir=out_dir,
        x_metric=args.plot_x_metric,
        y_metric=args.plot_y_metric,
        title=args.plot_title,
        label_points=args.label_points,
    )

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_dir": str(repo_dir),
                "python_exec": args.python_exec,
                "model_paths": [str(p) for p in model_paths],
                "dataset": args.dataset,
                "num_samples": args.num_samples,
                "sample_gpu": args.sample_gpu,
                "seeds": args.seeds,
                "plot_x_metric": args.plot_x_metric,
                "plot_y_metric": args.plot_y_metric,
                "label_points": args.label_points,
                "eval_args": eval_args,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDone.")
    print(f"long csv   : {out_dir / 'summary_long.csv'}")
    print(f"agg csv    : {out_dir / 'summary_agg.csv'}")
    print(f"pareto csv : {pareto_csv}")
    print(f"pareto png : {png_path}")
    print(f"pareto pdf : {pdf_path}")
    print(f"raw logs   : {raw_dir}")


if __name__ == "__main__":
    main()
