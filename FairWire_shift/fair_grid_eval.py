#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower().strip()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool from: {x}")


def sanitize_float(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def dataset_filename_tag(dataset: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset.strip())


def pareto_filename(dataset: str) -> str:
    dataset_tag = dataset_filename_tag(dataset)
    return f"pareto_curve_{dataset_tag}.jpg" if dataset_tag else "pareto_curve.jpg"


def summary_filename(dataset: str) -> str:
    dataset_tag = dataset_filename_tag(dataset)
    return f"summary_long_{dataset_tag}.csv" if dataset_tag else "summary_long.csv"


def format_plot_title(title: str, dataset: str) -> str:
    if not dataset:
        return title
    if "{dataset}" in title:
        return title.format(dataset=dataset)
    if dataset.lower() in title.lower():
        return title
    return f"{dataset}: {title}"


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_value(value: str) -> Any:
    if value is None:
        return value
    text = value.strip()
    if text == "":
        return text
    if text.lower() == "nan":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return value


def read_single_row_csv(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly 1 row in {path}, got {len(rows)}")
    return {key: parse_csv_value(value) for key, value in rows[0].items()}


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            {key: parse_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def make_combos(etas: List[float], ks: List[float], pair_mode: bool) -> List[Tuple[float, float]]:
    if pair_mode:
        if len(etas) != len(ks):
            raise ValueError("--pair_mode requires len(eta_values) == len(k_values)")
        return list(zip(etas, ks))
    return list(itertools.product(etas, ks))


def pick_metric(row: Dict[str, Any], candidates: List[str]) -> Tuple[str, float]:
    for key in candidates:
        if key in row:
            value = row[key]
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                return key, float(value)
    raise KeyError(f"None of metric keys found: {candidates}")


def aggregate_seed_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["eta"], row["k"])].append(row)

    agg_rows: List[Dict[str, Any]] = []
    for (eta, k), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        out: Dict[str, Any] = {
            "eta": eta,
            "k": k,
            "num_seeds": len(items),
        }
        numeric_keys = set()
        for item in items:
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    numeric_keys.add(key)
        for key in sorted(numeric_keys):
            vals = np.asarray([float(item[key]) for item in items if key in item], dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            out[f"{key}_mean"] = float(finite_vals.mean()) if finite_vals.size else float("nan")
            out[f"{key}_std"] = float(finite_vals.std()) if finite_vals.size else float("nan")
        agg_rows.append(out)
    return agg_rows


def pareto_front(rows: List[Dict[str, Any]], x_key: str, y_key: str) -> List[Dict[str, Any]]:
    out = []
    for i, a in enumerate(rows):
        ax = a.get(x_key, float("nan"))
        ay = a.get(y_key, float("nan"))
        if not np.isfinite(ax) or not np.isfinite(ay):
            continue
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            bx = b.get(x_key, float("nan"))
            by = b.get(y_key, float("nan"))
            if not np.isfinite(bx) or not np.isfinite(by):
                continue
            if (bx <= ax and by >= ay) and (bx < ax or by > ay):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return sorted(out, key=lambda row: (row[x_key], -row[y_key]))


def plot_pareto(
    agg_rows: List[Dict[str, Any]],
    front_rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    xerr_key: str,
    yerr_key: str,
    title: str,
    jpg_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))

    for row in agg_rows:
        x = row.get(x_key, float("nan"))
        y = row.get(y_key, float("nan"))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xerr = row.get(xerr_key, 0.0)
        yerr = row.get(yerr_key, 0.0)
        if not np.isfinite(xerr):
            xerr = 0.0
        if not np.isfinite(yerr):
            yerr = 0.0

        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", alpha=0.75, capsize=2)
        plt.annotate(
            f"({row['eta']}, {row['k']})",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    front_x = []
    front_y = []
    for row in front_rows:
        x = row.get(x_key, float("nan"))
        y = row.get(y_key, float("nan"))
        if np.isfinite(x) and np.isfinite(y):
            front_x.append(x)
            front_y.append(y)
    if front_x:
        order = np.argsort(front_x)
        front_x = np.asarray(front_x)[order]
        front_y = np.asarray(front_y)[order]
        plt.plot(front_x, front_y, linewidth=2)

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(jpg_path, dpi=200)
    plt.close()


def run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> int:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("STDOUT:\n")
        f.write(proc.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(proc.stderr)
        f.write(f"\n\nRETURNCODE: {proc.returncode}\n")
    return proc.returncode


def gpu_id_from_device(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    if device == "cuda":
        return 0
    return 0


def resolve_model_path(args, repo_dir: Path) -> Path:
    if args.model_path is not None:
        path = Path(args.model_path).expanduser()
        return path if path.is_absolute() else repo_dir / path
    return repo_dir / f"{args.dataset}_0.0_0.0_cpts" / "Sync_T3.pth"


def build_sample_cmd(args, model_path: Path, graph_path: Path, eta: float, k: float, seed: int) -> List[str]:
    cmd = [
        args.python_exec,
        "sample.py",
        "--model_path", str(model_path),
        "--num_samples", str(args.num_samples),
        "--gpu", str(gpu_id_from_device(args.gen_device)),
        "--seed", str(seed),
        "--save_pt_path", str(graph_path),
        "--skip_internal_eval",
    ]
    if eta > 0.0 and k > 0.0:
        cmd.extend([
            "--sp_shift",
            "--sp_eta", str(eta),
            "--sp_k", str(k),
            "--sp_eta_schedule", args.sp_eta_schedule,
        ])
        if args.sp_shift_clip is not None:
            cmd.extend(["--sp_shift_clip", str(args.sp_shift_clip)])
        if args.sp_guidance_normalize:
            cmd.append("--sp_guidance_normalize")
    return cmd


def build_generated_eval_cmd(args, graph_path: Path, per_graph_csv: Path, summary_csv: Path) -> List[str]:
    cmd = [
        args.python_exec,
        "evaluate_generated_graphs.py",
        "--graph_path", str(graph_path),
        "--dataset", args.dataset,
        "--label_attr", args.fair_sensitive_attr,
        "--sensitive_attr", args.fair_sensitive_attr,
        "--edge_sensitive_mode", args.fair_edge_sensitive_mode,
        "--device", args.lp_device,
        "--lp_model", args.lp_model,
        "--lp_epochs", str(args.lp_epochs),
        "--lp_test_ratio", str(args.lp_test_ratio),
        "--out_per_graph_csv", str(per_graph_csv),
        "--out_summary_csv", str(summary_csv),
    ]
    if args.fair_sensitive_value is not None:
        cmd.extend(["--sensitive_value", str(args.fair_sensitive_value)])
    if args.lp_hidden_dim is not None:
        cmd.extend(["--lp_hidden_dim", str(args.lp_hidden_dim)])
    if args.lp_out_dim is not None:
        cmd.extend(["--lp_out_dim", str(args.lp_out_dim)])
    if args.lp_dropout is not None:
        cmd.extend(["--lp_dropout", str(args.lp_dropout)])
    if args.lp_lr is not None:
        cmd.extend(["--lp_lr", str(args.lp_lr)])
    if args.lp_weight_decay is not None:
        cmd.extend(["--lp_weight_decay", str(args.lp_weight_decay)])
    if args.gat_heads is not None:
        cmd.extend(["--gat_heads", str(args.gat_heads)])
    return cmd


def draw_summary_pareto(per_run_rows: List[Dict[str, Any]], args, out_dir: Path) -> Path:
    agg_rows = aggregate_seed_rows(per_run_rows)
    x_key = "selected_sp_mean"
    y_key = "selected_auc_mean"
    xerr_key = "selected_sp_std"
    yerr_key = "selected_auc_std"
    front_rows = pareto_front(agg_rows, x_key=x_key, y_key=y_key)

    write_csv(agg_rows, out_dir / "aggregated_results.csv")
    write_csv(front_rows, out_dir / "pareto_front.csv")

    plot_path = out_dir / pareto_filename(args.dataset)
    plot_pareto(
        agg_rows=agg_rows,
        front_rows=front_rows,
        x_key=x_key,
        y_key=y_key,
        xerr_key=xerr_key,
        yerr_key=yerr_key,
        title=format_plot_title(args.plot_title, args.dataset),
        jpg_path=plot_path,
    )

    generic_path = out_dir / "pareto_curve.jpg"
    if generic_path != plot_path:
        plot_pareto(
            agg_rows=agg_rows,
            front_rows=front_rows,
            x_key=x_key,
            y_key=y_key,
            xerr_key=xerr_key,
            yerr_key=yerr_key,
            title=format_plot_title(args.plot_title, args.dataset),
            jpg_path=generic_path,
        )
    return plot_path


def parse_args():
    p = argparse.ArgumentParser(
        description="FairWire_shift eta/k logit-shift grid with generated-graph LP evaluation."
    )
    p.add_argument("--repo_dir", type=str, default=".")
    p.add_argument("--run_name", type=str, default=None, help="Accepted for EDGE_fairness CLI compatibility; FairWire uses --model_path or dataset_0.0_0.0_cpts/Sync_T3.pth.")
    p.add_argument("--checkpoint", type=int, default=None, help="Accepted for EDGE_fairness CLI compatibility; FairWire uses Sync_T3.pth.")
    p.add_argument("--model_path", type=str, default=None, help="Optional FairWire checkpoint path. Default: {dataset}_0.0_0.0_cpts/Sync_T3.pth")
    p.add_argument("--python_exec", type=str, default=sys.executable)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--eta_values", type=float, nargs="+", required=True)
    p.add_argument("--k_values", type=float, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--pair_mode", action="store_true")
    p.add_argument("--include_baseline", action="store_true")
    p.add_argument("--baseline_k", type=float, default=1.0)

    p.add_argument("--gen_device", type=str, default="cuda:0")
    p.add_argument("--lp_device", type=str, default="cuda:0")
    p.add_argument("--fair_sensitive_attr", type=str, default="y")
    p.add_argument("--fair_sensitive_value", type=int, default=None)
    p.add_argument("--fair_edge_sensitive_mode", type=str, default="either", choices=["either", "both"])
    p.add_argument("--largest_cc", type=str2bool, default=False, help="Accepted for CLI compatibility. FairWire_shift currently evaluates the full saved PyG graphs.")
    p.add_argument("--graph_variant", type=str, default="full", choices=["full", "eval"], help="Accepted for CLI compatibility. FairWire_shift saves one full PyG graph list.")

    p.add_argument("--sp_eta_schedule", choices=["constant", "early", "late"], default="constant")
    p.add_argument("--sp_shift_clip", type=float, default=None)
    p.add_argument(
        "--sp_guidance_normalize",
        action="store_true",
        help="Normalize SP raw guidance by graph-level mean(abs(raw guidance)) at each reverse step."
    )
    p.add_argument(
        "--fair_score_guidance_normalize",
        dest="sp_guidance_normalize",
        action="store_true",
        help="Alias for --sp_guidance_normalize."
    )

    p.add_argument("--lp_model", type=str, default="gcn")
    p.add_argument("--lp_epochs", type=int, default=200)
    p.add_argument("--lp_hidden_dim", type=int, default=None)
    p.add_argument("--lp_out_dim", type=int, default=None)
    p.add_argument("--lp_dropout", type=float, default=None)
    p.add_argument("--lp_lr", type=float, default=None)
    p.add_argument("--lp_weight_decay", type=float, default=None)
    p.add_argument("--lp_test_ratio", type=float, default=0.2)
    p.add_argument("--lp_val_ratio", dest="lp_test_ratio", type=float, help=argparse.SUPPRESS)
    p.add_argument("--gat_heads", type=int, default=None)

    p.add_argument("--auc_candidates", type=str, nargs="+", default=["lp/auc_mean", "aggregate_lp/auc"])
    p.add_argument(
        "--sp_candidates",
        type=str,
        nargs="+",
        default=[
            "lp/score_sp_abs_gap_mean",
            "aggregate_lp/score_sp_abs_gap",
            "lp/sp_abs_gap_mean",
            "aggregate_lp/sp_abs_gap",
        ],
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plot_title", type=str, default="LP Pareto: AUC vs SP")
    p.add_argument("--summary_csv", type=str, default=None, help="Draw Pareto from an existing summary_long csv and skip generation/evaluation.")
    return p.parse_args()


def main():
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_csv is not None:
        per_run_rows = read_csv_rows(Path(args.summary_csv).expanduser().resolve())
        per_run_rows = [row for row in per_run_rows if str(row.get("dataset", args.dataset)) == args.dataset]
        plot_path = draw_summary_pareto(per_run_rows, args, out_dir)
        print(f"summary csv: {args.summary_csv}")
        print(f"pareto jpg : {plot_path}")
        return

    if not (repo_dir / "sample.py").exists():
        raise FileNotFoundError(f"sample.py not found in repo_dir: {repo_dir}")
    if not (repo_dir / "evaluate_generated_graphs.py").exists():
        raise FileNotFoundError(f"evaluate_generated_graphs.py not found in repo_dir: {repo_dir}")
    if not (repo_dir / "graphs" / f"{args.dataset}_feat.pkl").exists():
        raise FileNotFoundError(f"Missing reference graph: {repo_dir / 'graphs' / f'{args.dataset}_feat.pkl'}")

    model_path = resolve_model_path(args, repo_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing FairWire checkpoint: {model_path}")

    combos = make_combos(args.eta_values, args.k_values, args.pair_mode)
    if args.include_baseline:
        combos = [(0.0, args.baseline_k)] + [c for c in combos if not (c[0] == 0.0 and c[1] == args.baseline_k)]

    raw_log_dir = out_dir / "raw_logs"
    gen_dir = out_dir / "generated_graphs"
    eval_dir = out_dir / "evaluated_graphs"
    gen_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows: List[Dict[str, Any]] = []
    total = len(combos) * len(args.seeds)
    step = 0
    print(f"Model path : {model_path}")
    print(f"Combos     : {len(combos)}")
    print(f"Seeds      : {args.seeds}")
    print(f"Total runs : {total}")

    for eta, k in combos:
        for seed in args.seeds:
            step += 1
            tag = f"eta_{sanitize_float(eta)}_k_{sanitize_float(k)}_seed_{seed}"
            graph_path = gen_dir / f"{tag}.pyg.pt"
            run_eval_dir = eval_dir / tag
            run_eval_dir.mkdir(parents=True, exist_ok=True)
            per_graph_csv = run_eval_dir / "per_graph.csv"
            summary_csv = run_eval_dir / "summary.csv"
            gen_log = raw_log_dir / f"{tag}.generate.txt"
            eval_log = raw_log_dir / f"{tag}.eval_generated.txt"

            print(f"[{step}/{total}] eta={eta}, k={k}, seed={seed}")

            generate_returncode = 0
            if not (args.skip_existing and graph_path.exists()):
                generate_cmd = build_sample_cmd(args, model_path, graph_path, eta, k, seed)
                generate_returncode = run_cmd(generate_cmd, repo_dir, gen_log)

            row_base = {
                "dataset": args.dataset,
                "eta": eta,
                "k": k,
                "seed": seed,
                "generate_returncode": generate_returncode,
                "generated_eval_returncode": float("nan"),
                "graph_path": str(graph_path),
                "summary_csv": str(summary_csv),
                "generate_log": str(gen_log),
                "generated_eval_log": str(eval_log),
            }
            if generate_returncode != 0:
                per_run_rows.append({**row_base, "error": "generation_failed"})
                continue

            eval_returncode = 0
            if not (args.skip_existing and summary_csv.exists()):
                eval_cmd = build_generated_eval_cmd(args, graph_path, per_graph_csv, summary_csv)
                eval_returncode = run_cmd(eval_cmd, repo_dir, eval_log)

            if eval_returncode != 0:
                per_run_rows.append({**row_base, "generated_eval_returncode": eval_returncode, "error": "generated_eval_failed"})
                continue

            summary_row = read_single_row_csv(summary_csv)
            try:
                auc_key, auc_val = pick_metric(summary_row, args.auc_candidates)
            except Exception:
                auc_key, auc_val = "NA", float("nan")
            try:
                sp_key, sp_val = pick_metric(summary_row, args.sp_candidates)
            except Exception:
                sp_key, sp_val = "NA", float("nan")

            row = {
                **row_base,
                "generated_eval_returncode": eval_returncode,
                "selected_auc_key": auc_key,
                "selected_auc": auc_val,
                "selected_sp_key": sp_key,
                "selected_sp": sp_val,
            }
            row.update(summary_row)
            row["dataset"] = args.dataset
            per_run_rows.append(row)

    summary_path = out_dir / "summary_long.csv"
    dataset_summary_path = out_dir / summary_filename(args.dataset)
    write_csv(per_run_rows, summary_path)
    if dataset_summary_path != summary_path:
        write_csv(per_run_rows, dataset_summary_path)

    if not any(row.get("generate_returncode") == 0 and row.get("generated_eval_returncode") == 0 for row in per_run_rows):
        raise RuntimeError(f"All grid runs failed. See {summary_path} and {raw_log_dir}")

    plot_path = draw_summary_pareto(per_run_rows, args, out_dir)
    meta = vars(args).copy()
    meta.update({"model_path": str(model_path), "combos": combos})
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"summary csv: {summary_path}")
    if dataset_summary_path != summary_path:
        print(f"dataset csv: {dataset_summary_path}")
    print(f"pareto jpg : {plot_path}")
    generic_plot = out_dir / "pareto_curve.jpg"
    if generic_plot != plot_path:
        print(f"pareto jpg : {generic_plot}")


if __name__ == "__main__":
    main()
