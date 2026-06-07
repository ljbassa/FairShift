#!/usr/bin/env python3
import argparse
import csv
import itertools
import math
import os
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
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {key: parse_csv_value(value) for key, value in row.items()}
            for row in reader
        ]


def normalize_dataset(value: Any) -> str:
    return str(value).strip().lower()


def select_dataset_rows(
    rows: List[Dict[str, Any]],
    dataset: str,
    source: Path,
) -> List[Dict[str, Any]]:
    if not any("dataset" in row for row in rows):
        inferred_dataset = infer_dataset_from_summary_path(source)
        if inferred_dataset is not None and normalize_dataset(inferred_dataset) == normalize_dataset(dataset):
            for row in rows:
                row["dataset"] = dataset
            return rows

        raise RuntimeError(
            f"{source} has no 'dataset' column, so --dataset {dataset!r} cannot be verified. "
            "This summary was likely created by an older script. Re-run the grid for each dataset "
            f"to create a dataset-labeled summary_long.csv, or use summary_long_{dataset}.csv "
            "when the dataset can be inferred from the file name."
        )

    target = normalize_dataset(dataset)
    filtered = [row for row in rows if normalize_dataset(row.get("dataset", "")) == target]
    if not filtered:
        available = sorted({str(row.get("dataset", "")).strip() for row in rows if str(row.get("dataset", "")).strip()})
        raise RuntimeError(
            f"{source} contains no rows for dataset={dataset!r}. "
            f"Available datasets: {', '.join(available) if available else 'none'}"
        )
    return filtered


def format_plot_title(title: str, dataset: str) -> str:
    if not dataset:
        return title
    if "{dataset}" in title:
        return title.format(dataset=dataset)
    if dataset.lower() in title.lower():
        return title
    return f"{dataset}: {title}"


def dataset_filename_tag(dataset: str) -> str:
    dataset_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset.strip())
    return dataset_tag


def pareto_filename(dataset: str) -> str:
    dataset_tag = dataset_filename_tag(dataset)
    return f"pareto_curve_{dataset_tag}.jpg" if dataset_tag else "pareto_curve.jpg"


def summary_filename(dataset: str) -> str:
    dataset_tag = dataset_filename_tag(dataset)
    return f"summary_long_{dataset_tag}.csv" if dataset_tag else "summary_long.csv"


def infer_dataset_from_summary_path(path: Path) -> Optional[str]:
    stem = path.stem
    prefix = "summary_long_"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix):]
    return None


def aggregate_seed_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["eta"], row["k"])].append(row)

    agg_rows: List[Dict[str, Any]] = []
    for (eta, k), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
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
            vals = np.asarray([float(it[key]) for it in items if key in it], dtype=float)
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
    png_path: Path,
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

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=200)
    plt.close()


def build_generate_args(args, eta: float, k: float, seed: int) -> argparse.Namespace:
    from evaluate import build_parser as build_generate_parser

    parser = build_generate_parser()
    argv = [
        "--dataset", args.dataset,
        "--run_name", args.run_name,
        "--checkpoint", str(args.checkpoint),
        "--num_samples", str(args.num_samples),
        "--seed", str(seed),
        "--device", args.gen_device,
        "--fair_score_sp",
        "--fair_score_eta", str(eta),
        "--fair_score_k", str(k),
        "--fair_sensitive_attr", args.fair_sensitive_attr,
        "--fair_edge_sensitive_mode", args.fair_edge_sensitive_mode,
        "--largest_cc", str(args.largest_cc),
    ]
    if args.fair_score_guidance_normalize:
        argv += ["--fair_score_guidance_normalize", "True"]
    if args.fair_sensitive_value is not None:
        argv += ["--fair_sensitive_value", str(args.fair_sensitive_value)]

    gen_args = parser.parse_args(argv)
    gen_args.collect_generated_graphs = True
    gen_args.save_samples = False
    gen_args.save_full_graph = False
    if args.run_dir is not None:
        gen_args.run_dir = str(Path(args.run_dir).expanduser().resolve())
    return gen_args


def resolve_run_dir(args, repo_dir: Path) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser().resolve()
    return repo_dir / "wandb" / args.dataset / "multinomial_diffusion" / "multistep" / args.run_name


def validate_inputs(args, repo_dir: Path) -> None:
    run_dir = resolve_run_dir(args, repo_dir)
    required_paths = [
        run_dir / "args.pickle",
        run_dir / "check" / f"checkpoint_{args.checkpoint - 1}.pt",
        repo_dir / "graphs" / f"{args.dataset}_feat.pkl",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required input files before evaluation starts:\n"
            f"{missing_text}\n"
            "Set --repo_dir to the repo that contains the trained run, or pass "
            "--run_dir explicitly when the trained run lives somewhere else."
        )


def build_generated_eval_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        graph_path=None,
        dataset=args.dataset,
        graph_index=None,
        seed=0,
        label_attr=args.fair_sensitive_attr,
        sensitive_attr=args.fair_sensitive_attr,
        sensitive_value=args.fair_sensitive_value,
        edge_sensitive_mode=args.fair_edge_sensitive_mode,
        max_pos_edges=20000,
        neg_ratio=1.0,
        lp_model=args.lp_model,
        lp_num_layers=2,
        lp_hidden_dim=args.lp_hidden_dim if args.lp_hidden_dim is not None else 128,
        lp_out_dim=args.lp_out_dim if args.lp_out_dim is not None else 64,
        lp_dropout=args.lp_dropout if args.lp_dropout is not None else 0.1,
        lp_lr=args.lp_lr if args.lp_lr is not None else 1e-2,
        lp_weight_decay=args.lp_weight_decay if args.lp_weight_decay is not None else 0.0,
        lp_epochs=args.lp_epochs,
        lp_patience=30,
        lp_batch_size=16384,
        lp_test_ratio=args.lp_test_ratio,
        gat_heads=args.gat_heads if args.gat_heads is not None else 4,
        device=args.lp_device,
        threshold=0.5,
        lp_search=False,
        lp_search_hidden_dims=[64, 128],
        lp_search_lrs=[1e-2, 3e-3],
        lp_search_dropouts=[0.0, 0.1, 0.2],
        lp_search_num_layers=[1, 2],
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Grid search fairness guidance and evaluate generated graphs without storing intermediates."
    )
    p.add_argument("--repo_dir", type=str, default=".")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Optional explicit trained-run directory. Use this if wandb outputs live outside --repo_dir.",
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=64)

    p.add_argument("--eta_values", type=float, nargs="+", default=None)
    p.add_argument("--k_values", type=float, nargs="+", default=None)
    p.add_argument("--fair_score_guidance_normalize", type=eval, default=False)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--pair_mode", action="store_true")
    p.add_argument("--include_baseline", action="store_true")
    p.add_argument("--baseline_k", type=float, default=1.0)

    p.add_argument("--gen_device", type=str, default="cuda:0")
    p.add_argument("--lp_device", type=str, default="cuda:0")

    p.add_argument("--fair_sensitive_attr", type=str, default="y")
    p.add_argument("--fair_sensitive_value", type=int, default=None)
    p.add_argument("--fair_edge_sensitive_mode", type=str, default="either", choices=["either", "both"])

    p.add_argument("--largest_cc", type=str2bool, default=False)
    p.add_argument(
        "--graph_variant",
        type=str,
        default="full",
        choices=["full", "eval"],
        help="Evaluate in-memory full sampled PyG graphs or the eval graphs after optional largest-CC filtering.",
    )

    p.add_argument("--lp_model", type=str, default="gcn", choices=["gcn", "sage", "gat"])
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
            "lp/sp_abs_gap_mean",
            "lp/score_sp_abs_gap_mean",
            "aggregate_lp/sp_abs_gap",
            "aggregate_lp/score_sp_abs_gap",
        ],
    )

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--plot_title", type=str, default="LP Pareto: AUC vs SP")
    p.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help=(
            "Draw pareto_curve_{dataset}.jpg from an existing dataset-labeled "
            "summary_long.csv or summary_long_{dataset}.csv and skip generation/evaluation."
        ),
    )
    args = p.parse_args()

    if args.summary_csv is None:
        missing = []
        if args.run_name is None:
            missing.append("--run_name")
        if args.checkpoint is None:
            missing.append("--checkpoint")
        if args.eta_values is None:
            missing.append("--eta_values")
        if args.k_values is None:
            missing.append("--k_values")
        if missing:
            p.error("the following arguments are required unless --summary_csv is provided: " + ", ".join(missing))

    return args


def draw_summary_pareto(per_run_rows: List[Dict[str, Any]], args, out_dir: Path) -> Path:
    agg_rows = aggregate_seed_rows(per_run_rows)
    x_key = "selected_sp_mean"
    y_key = "selected_auc_mean"
    xerr_key = "selected_sp_std"
    yerr_key = "selected_auc_std"

    front_rows = pareto_front(agg_rows, x_key=x_key, y_key=y_key)
    plot_path = out_dir / pareto_filename(args.dataset)
    plot_pareto(
        agg_rows=agg_rows,
        front_rows=front_rows,
        x_key=x_key,
        y_key=y_key,
        xerr_key=xerr_key,
        yerr_key=yerr_key,
        title=format_plot_title(args.plot_title, args.dataset),
        png_path=plot_path,
    )
    return plot_path


def main():
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_csv is not None:
        summary_csv = Path(args.summary_csv).expanduser().resolve()
        all_rows = read_csv_rows(summary_csv)
        if not all_rows:
            raise RuntimeError(f"No rows found in {summary_csv}")
        per_run_rows = select_dataset_rows(all_rows, args.dataset, summary_csv)
        plot_path = draw_summary_pareto(per_run_rows, args, out_dir)
        print(f"summary csv: {summary_csv}")
        print(f"dataset    : {args.dataset} ({len(per_run_rows)} rows)")
        print(f"pareto jpg : {plot_path}")
        return

    validate_inputs(args, repo_dir)

    combos = make_combos(args.eta_values, args.k_values, args.pair_mode)
    if args.include_baseline:
        combos = [(0.0, args.baseline_k)] + [c for c in combos if not (c[0] == 0.0 and c[1] == args.baseline_k)]

    per_run_rows: List[Dict[str, Any]] = []
    total = len(combos) * len(args.seeds)
    step = 0
    from evaluate import run_evaluate
    from evaluate_generated_graphs import evaluate_graphs as evaluate_generated_graphs

    prev_cwd = Path.cwd()
    os.chdir(repo_dir)
    try:
        reference_graph_path = str(repo_dir / "graphs" / f"{args.dataset}_feat.pkl")
        for eta, k in combos:
            for seed in args.seeds:
                step += 1
                print(f"[{step}/{total}] eta={eta}, k={k}, seed={seed}")

                try:
                    generate_args = build_generate_args(args, eta, k, seed)
                    gen_result = run_evaluate(generate_args)
                    graphs = gen_result["pyg_full_datas"] if args.graph_variant == "full" else gen_result["pyg_eval_datas"]

                    eval_args = build_generated_eval_args(args)
                    _per_graph_rows, summary_row = evaluate_generated_graphs(
                        graphs=graphs,
                        args=eval_args,
                        reference_graph_path=reference_graph_path,
                        total_loaded=len(graphs),
                    )

                    try:
                        auc_key, auc_val = pick_metric(summary_row, args.auc_candidates)
                    except Exception:
                        auc_key, auc_val = "NA", float("nan")

                    try:
                        sp_key, sp_val = pick_metric(summary_row, args.sp_candidates)
                    except Exception:
                        sp_key, sp_val = "NA", float("nan")

                    row = {
                        "dataset": args.dataset,
                        "eta": eta,
                        "k": k,
                        "seed": seed,
                        "generate_returncode": 0,
                        "generated_eval_returncode": 0,
                        "selected_auc_key": auc_key,
                        "selected_auc": auc_val,
                        "selected_sp_key": sp_key,
                        "selected_sp": sp_val,
                    }
                    row.update(summary_row)
                    row["dataset"] = args.dataset
                    per_run_rows.append(row)
                except Exception as exc:
                    per_run_rows.append({
                        "dataset": args.dataset,
                        "eta": eta,
                        "k": k,
                        "seed": seed,
                        "generate_returncode": 1,
                        "generated_eval_returncode": 1,
                        "selected_auc_key": "NA",
                        "selected_auc": float("nan"),
                        "selected_sp_key": "NA",
                        "selected_sp": float("nan"),
                        "error": str(exc),
                    })
    finally:
        os.chdir(prev_cwd)

    summary_path = out_dir / "summary_long.csv"
    dataset_summary_path = out_dir / summary_filename(args.dataset)
    write_csv(per_run_rows, summary_path)
    if dataset_summary_path != summary_path:
        write_csv(per_run_rows, dataset_summary_path)
    if not any(row.get("generate_returncode") == 0 and row.get("generated_eval_returncode") == 0 for row in per_run_rows):
        raise RuntimeError(f"All grid runs failed. See {summary_path} for captured errors.")

    plot_path = draw_summary_pareto(per_run_rows, args, out_dir)
    print(f"summary csv: {summary_path}")
    if dataset_summary_path != summary_path:
        print(f"dataset csv: {dataset_summary_path}")
    print(f"pareto jpg : {plot_path}")


if __name__ == "__main__":
    main()
