#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Discover FairWire checkpoints trained with different -aA values, "
            "run sample.py -> evaluate_generated_graphs.py via fair_grid_eval.py, "
            "and print/export LP AUC vs score-SP metrics."
        )
    )
    parser.add_argument("--repo_dir", type=Path, default=Path.cwd())
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--alphaX", type=float, default=0.0)
    parser.add_argument(
        "--aA_values",
        type=float,
        nargs="*",
        default=None,
        help="Optional explicit aA values. If omitted, discover all matching checkpoints.",
    )
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--sample_gpu", type=int, default=0)
    parser.add_argument(
        "--eval_device",
        default=None,
        help="Device for evaluate_generated_graphs.py. Defaults to cuda:<sample_gpu>.",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--label_points", choices=["none", "front", "all"], default="all")
    parser.add_argument("--plot_title", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Extra evaluate_generated_graphs.py args after '--'.",
    )
    args = parser.parse_args()
    if args.eval_args and args.eval_args[0] == "--":
        args.eval_args = args.eval_args[1:]
    return args


def tag_float(value):
    return str(float(value))


def parse_alpha_a(path, dataset, alpha_x):
    pattern = re.compile(
        rf"^{re.escape(dataset)}_(?P<aA>[-+0-9.eE]+)_{re.escape(tag_float(alpha_x))}_cpts$"
    )
    match = pattern.match(path.parent.name)
    if not match:
        return None
    try:
        return float(match.group("aA"))
    except ValueError:
        return None


def discover_checkpoints(repo_dir, dataset, T, alpha_x, aA_values):
    if aA_values is not None and len(aA_values) > 0:
        paths = [
            repo_dir / f"{dataset}_{tag_float(aA)}_{tag_float(alpha_x)}_cpts" / f"Sync_T{T}.pth"
            for aA in aA_values
        ]
    else:
        paths = sorted(repo_dir.glob(f"{dataset}_*_{tag_float(alpha_x)}_cpts/Sync_T{T}.pth"))

    rows = []
    for path in paths:
        alpha_a = parse_alpha_a(path, dataset, alpha_x)
        if alpha_a is None and aA_values is not None:
            alpha_a = float(path.parent.name.split("_")[1])
        rows.append((alpha_a, path))

    missing = [str(path) for _alpha_a, path in rows if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))

    rows = [(alpha_a, path) for alpha_a, path in rows if path.exists()]
    rows.sort(key=lambda item: (float("inf") if item[0] is None else item[0], str(item[1])))
    if not rows:
        raise FileNotFoundError(
            f"No checkpoints found for pattern: {dataset}_*_{tag_float(alpha_x)}_cpts/Sync_T{T}.pth"
        )
    return rows


def build_grid_command(args, repo_dir, checkpoints):
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = repo_dir / f"fairwire_aA_pareto_{args.dataset}_T{args.T}"
    elif not out_dir.is_absolute():
        out_dir = repo_dir / out_dir

    eval_device = args.eval_device or f"cuda:{args.sample_gpu}"
    plot_title = args.plot_title or f"{args.dataset} FairWire aA Pareto T{args.T}"

    cmd = [
        args.python_exec,
        "fair_grid_eval.py",
        "--repo_dir",
        str(repo_dir),
        "--python_exec",
        args.python_exec,
        "--dataset",
        args.dataset,
        "--model_paths",
        *[str(path) for _alpha_a, path in checkpoints],
        "--num_samples",
        str(args.num_samples),
        "--sample_gpu",
        str(args.sample_gpu),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--out_dir",
        str(out_dir),
        "--label_points",
        args.label_points,
        "--plot_title",
        plot_title,
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")

    cmd += ["--", "--device", eval_device, *args.eval_args]
    return cmd, out_dir


def read_metrics(summary_csv):
    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    def alpha_from_model_tag(row):
        tag = row.get("model_tag", "")
        match = re.search(r"_([-+0-9.eE]+)_0\.0_cpts__Sync_T", tag)
        if not match:
            return float("inf")
        try:
            return float(match.group(1))
        except ValueError:
            return float("inf")

    rows.sort(key=alpha_from_model_tag)
    return rows


def write_metric_summary(rows, out_path):
    fields = [
        "aA",
        "model_tag",
        "n_runs",
        "n_success",
        "seeds",
        "lp/auc_mean",
        "lp/auc_mean_std",
        "lp/score_sp_abs_gap_mean",
        "lp/score_sp_abs_gap_mean_std",
        "lp/score_sp_gap_mean",
        "lp/score_sp_gap_mean_std",
        "aggregate_lp/auc",
        "aggregate_lp/score_sp_abs_gap",
        "aggregate_lp/score_sp_gap",
    ]

    def alpha_from_model_tag(row):
        tag = row.get("model_tag", "")
        match = re.search(r"_([-+0-9.eE]+)_0\.0_cpts__Sync_T", tag)
        return match.group(1) if match else ""

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["aA"] = alpha_from_model_tag(row)
            writer.writerow(out)


def print_metrics(rows):
    fields = [
        ("aA", "aA"),
        ("lp/auc_mean", "auc"),
        ("lp/score_sp_abs_gap_mean", "score_sp_abs"),
        ("lp/score_sp_gap_mean", "score_sp"),
        ("n_success", "ok"),
        ("seeds", "seeds"),
    ]

    table = []
    for row in rows:
        tag = row.get("model_tag", "")
        match = re.search(r"_([-+0-9.eE]+)_0\.0_cpts__Sync_T", tag)
        alpha_a = match.group(1) if match else "?"
        table.append({
            "aA": alpha_a,
            "auc": row.get("lp/auc_mean", ""),
            "score_sp_abs": row.get("lp/score_sp_abs_gap_mean", ""),
            "score_sp": row.get("lp/score_sp_gap_mean", ""),
            "ok": row.get("n_success", ""),
            "seeds": row.get("seeds", ""),
        })

    widths = {
        label: max(len(label), *(len(str(row[label])) for row in table))
        for _key, label in fields
    }
    print()
    print("LP AUC / score-SP summary")
    print("  " + "  ".join(label.rjust(widths[label]) for _key, label in fields))
    for row in table:
        print("  " + "  ".join(str(row[label]).rjust(widths[label]) for _key, label in fields))


def main():
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    checkpoints = discover_checkpoints(repo_dir, args.dataset, args.T, args.alphaX, args.aA_values)

    print("Discovered checkpoints:")
    for alpha_a, path in checkpoints:
        print(f"  aA={alpha_a:g}  {path.relative_to(repo_dir)}")

    cmd, out_dir = build_grid_command(args, repo_dir, checkpoints)
    print()
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    proc = subprocess.run(cmd, cwd=repo_dir)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    summary_csv = out_dir / "summary_agg.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Expected summary CSV not found: {summary_csv}")

    rows = read_metrics(summary_csv)
    metric_csv = out_dir / "metrics_auc_sp.csv"
    write_metric_summary(rows, metric_csv)
    print_metrics(rows)

    print()
    print(f"metric csv : {metric_csv}")
    print(f"summary csv: {summary_csv}")
    print(f"pareto csv : {out_dir / (args.dataset + '_pareto_front.csv')}")
    print(f"pareto png : {out_dir / (args.dataset + '_pareto_curve.png')}")


if __name__ == "__main__":
    main()
