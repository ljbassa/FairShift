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
from typing import Dict, List, Tuple, Any

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
    s = f"{x}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def try_number(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return v
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return ""
    if s.lower() == "nan":
        return float("nan")
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return s


def read_single_row_csv(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly 1 row in {path}, got {len(rows)}")
    row = {k: try_number(v) for k, v in rows[0].items()}
    return row


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

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


def find_saved_graph_file(save_dir: Path, variant: str) -> Path:
    suffix = ".pyg_full.pt" if variant == "full" else ".pyg.pt"
    cands = sorted(save_dir.rglob(f"*{suffix}"))
    if len(cands) == 0:
        raise FileNotFoundError(f"No saved graph file matching *{suffix} under {save_dir}")
    if len(cands) > 1:
        raise RuntimeError(
            f"Expected one saved graph file under {save_dir}, found {len(cands)}:\n" +
            "\n".join(str(x) for x in cands)
        )
    return cands[0]


def pick_metric(row: Dict[str, Any], candidates: List[str]) -> Tuple[str, float]:
    for k in candidates:
        if k in row:
            v = row[k]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                return k, float(v)
    raise KeyError(f"None of metric keys found: {candidates}")


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
            for kk, vv in item.items():
                if isinstance(vv, (int, float)):
                    numeric_keys.add(kk)

        for kk in sorted(numeric_keys):
            vals = np.asarray([float(it[kk]) for it in items if kk in it], dtype=float)
            out[f"{kk}_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
            out[f"{kk}_std"] = float(np.nanstd(vals)) if vals.size else float("nan")

        agg_rows.append(out)
    return agg_rows


def pareto_front(rows: List[Dict[str, Any]], x_key: str, y_key: str) -> List[Dict[str, Any]]:
    """
    Minimize x, maximize y.
    Non-dominated set.
    """
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

            # b dominates a if bx <= ax and by >= ay, and one strict
            if (bx <= ax and by >= ay) and (bx < ax or by > ay):
                dominated = True
                break
        if not dominated:
            out.append(a)

    out = sorted(out, key=lambda r: (r[x_key], -r[y_key]))
    return out


def plot_pareto(
    agg_rows: List[Dict[str, Any]],
    front_rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    xerr_key: str,
    yerr_key: str,
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))

    # all points
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

        plt.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt="o",
            alpha=0.75,
            capsize=2,
        )
        plt.annotate(
            f"({row['eta']}, {row['k']})",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    # pareto front line
    fx = []
    fy = []
    for row in front_rows:
        x = row.get(x_key, float("nan"))
        y = row.get(y_key, float("nan"))
        if np.isfinite(x) and np.isfinite(y):
            fx.append(x)
            fy.append(y)
    if fx:
        order = np.argsort(fx)
        fx = np.asarray(fx)[order]
        fy = np.asarray(fy)[order]
        plt.plot(fx, fy, linewidth=2)

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()


def build_generate_cmd(args, eta: float, k: float, seed: int, save_dir: Path) -> List[str]:
    generate_script = Path(args.generate_script)
    if not generate_script.is_absolute():
        generate_script = Path(args.repo_dir) / generate_script

    cmd = [
        args.python_exec,
        str(generate_script),
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
        "--save_samples",
        "--save_dir", str(save_dir),
        "--largest_cc", str(args.largest_cc),
    ]
    if args.fair_sensitive_value is not None:
        cmd += ["--fair_sensitive_value", str(args.fair_sensitive_value)]
    if args.save_full_graph:
        cmd += ["--save_full_graph"]
    return cmd


def build_generated_eval_cmd(
    args,
    graph_path: Path,
    per_graph_csv: Path,
    summary_csv: Path,
) -> List[str]:
    eval_script = Path(args.generated_eval_script)
    if not eval_script.is_absolute():
        eval_script = Path(args.repo_dir) / eval_script

    cmd = [
        args.python_exec,
        str(eval_script),
        "--graph_path", str(graph_path),
        "--dataset", args.dataset,
        "--sensitive_attr", args.fair_sensitive_attr,
        "--edge_sensitive_mode", args.fair_edge_sensitive_mode,
        "--device", args.lp_device,
        "--lp_model", args.lp_model,
        "--lp_epochs", str(args.lp_epochs),
        "--out_per_graph_csv", str(per_graph_csv),
        "--out_summary_csv", str(summary_csv),
    ]
    if args.fair_sensitive_value is not None:
        cmd += ["--sensitive_value", str(args.fair_sensitive_value)]

    # optional passthroughs
    if args.lp_hidden_dim is not None:
        cmd += ["--lp_hidden_dim", str(args.lp_hidden_dim)]
    if args.lp_out_dim is not None:
        cmd += ["--lp_out_dim", str(args.lp_out_dim)]
    if args.lp_dropout is not None:
        cmd += ["--lp_dropout", str(args.lp_dropout)]
    if args.lp_lr is not None:
        cmd += ["--lp_lr", str(args.lp_lr)]
    if args.lp_weight_decay is not None:
        cmd += ["--lp_weight_decay", str(args.lp_weight_decay)]
    if args.lp_test_ratio is not None:
        cmd += ["--lp_test_ratio", str(args.lp_test_ratio)]
    if args.gat_heads is not None:
        cmd += ["--gat_heads", str(args.gat_heads)]
    return cmd


def parse_args():
    p = argparse.ArgumentParser(
        description="Grid search fairness guidance with evaluate.py -> evaluate_generated_graphs.py pipeline."
    )

    # generation side
    p.add_argument("--repo_dir", type=str, required=True)
    p.add_argument("--python_exec", type=str, default=sys.executable)
    p.add_argument("--generate_script", type=str, default="evaluate.py")
    p.add_argument("--generated_eval_script", type=str, default="evaluate_generated_graphs.py")

    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=int, required=True)
    p.add_argument("--num_samples", type=int, default=64)

    p.add_argument("--eta_values", type=float, nargs="+", required=True)
    p.add_argument("--k_values", type=float, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])

    p.add_argument("--pair_mode", action="store_true")
    p.add_argument("--include_baseline", action="store_true")
    p.add_argument("--baseline_k", type=float, default=1.0)

    p.add_argument("--gen_device", type=str, default="cuda:0")
    p.add_argument("--lp_device", type=str, default="cuda:0")

    p.add_argument("--fair_sensitive_attr", type=str, default="y",
                   help="Node attribute used to define same-label vs different-label pair groups.")
    p.add_argument("--fair_sensitive_value", type=int, default=None,
                   help="Legacy no-op retained for backward compatibility.")
    p.add_argument("--fair_edge_sensitive_mode", type=str, default="either", choices=["either", "both"],
                   help="Legacy no-op retained for backward compatibility.")

    p.add_argument("--largest_cc", type=str2bool, default=False)
    p.add_argument("--save_full_graph", action="store_true")
    p.add_argument("--graph_variant", type=str, default="full", choices=["full", "eval"],
                   help="Which saved graph file to evaluate: *.pyg_full.pt or *.pyg.pt")

    # LP evaluator side
    p.add_argument("--lp_model", type=str, default="gcn", choices=["gcn", "sage", "gat"])
    p.add_argument("--lp_epochs", type=int, default=200)
    p.add_argument("--lp_hidden_dim", type=int, default=None)
    p.add_argument("--lp_out_dim", type=int, default=None)
    p.add_argument("--lp_dropout", type=float, default=None)
    p.add_argument("--lp_lr", type=float, default=None)
    p.add_argument("--lp_weight_decay", type=float, default=None)
    p.add_argument("--lp_test_ratio", type=float, default=None)
    p.add_argument("--gat_heads", type=int, default=None)

    # metric selection for pareto
    p.add_argument("--auc_candidates", type=str, nargs="+",
                   default=["lp/auc_mean", "aggregate_lp/auc"])
    p.add_argument("--sp_candidates", type=str, nargs="+",
                   default=["lp/sp_abs_gap_mean", "lp/score_sp_abs_gap_mean", "aggregate_lp/sp_abs_gap", "aggregate_lp/score_sp_abs_gap"])

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plot_title", type=str, default="LP Pareto: AUC vs SP")

    return p.parse_args()


def main():
    args = parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_log_dir = out_dir / "raw_logs"
    gen_dir = out_dir / "generated_graphs"
    eval_dir = out_dir / "evaluated_graphs"
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    combos = make_combos(args.eta_values, args.k_values, args.pair_mode)
    if args.include_baseline:
        combos = [(0.0, args.baseline_k)] + [c for c in combos if not (c[0] == 0.0 and c[1] == args.baseline_k)]

    per_run_rows: List[Dict[str, Any]] = []

    total = len(combos) * len(args.seeds)
    step = 0

    for eta, k in combos:
        for seed in args.seeds:
            step += 1
            tag = f"eta_{sanitize_float(eta)}_k_{sanitize_float(k)}_seed_{seed}"
            run_save_dir = gen_dir / tag
            run_eval_dir = eval_dir / tag
            run_eval_dir.mkdir(parents=True, exist_ok=True)

            gen_log = raw_log_dir / f"{tag}.generate.txt"
            eval_log = raw_log_dir / f"{tag}.eval_generated.txt"
            per_graph_csv = run_eval_dir / "per_graph.csv"
            summary_csv = run_eval_dir / "summary.csv"

            # -------------------------------------------------
            # 1) generate graphs with evaluate.py
            # -------------------------------------------------
            graph_path = None
            if args.skip_existing:
                try:
                    graph_path = find_saved_graph_file(run_save_dir, args.graph_variant)
                except Exception:
                    graph_path = None

            if graph_path is None:
                run_save_dir.mkdir(parents=True, exist_ok=True)
                gen_cmd = build_generate_cmd(args, eta, k, seed, run_save_dir)
                rc = run_cmd(gen_cmd, repo_dir, gen_log)
                if rc != 0:
                    per_run_rows.append({
                        "eta": eta,
                        "k": k,
                        "seed": seed,
                        "generate_returncode": rc,
                        "generated_eval_returncode": float("nan"),
                        "generate_log": str(gen_log),
                        "generated_eval_log": str(eval_log),
                        "graph_path": "",
                        "summary_csv": "",
                        "error": "generation_failed",
                    })
                    continue
                graph_path = find_saved_graph_file(run_save_dir, args.graph_variant)

            # -------------------------------------------------
            # 2) evaluate saved graphs with evaluate_generated_graphs.py
            # -------------------------------------------------
            if not (args.skip_existing and summary_csv.exists()):
                eval_cmd = build_generated_eval_cmd(
                    args=args,
                    graph_path=graph_path,
                    per_graph_csv=per_graph_csv,
                    summary_csv=summary_csv,
                )
                rc2 = run_cmd(eval_cmd, repo_dir, eval_log)
                if rc2 != 0:
                    per_run_rows.append({
                        "eta": eta,
                        "k": k,
                        "seed": seed,
                        "generate_returncode": 0,
                        "generated_eval_returncode": rc2,
                        "generate_log": str(gen_log),
                        "generated_eval_log": str(eval_log),
                        "graph_path": str(graph_path),
                        "summary_csv": str(summary_csv),
                        "error": "generated_eval_failed",
                    })
                    continue

            # -------------------------------------------------
            # 3) parse summary csv and collect metrics
            # -------------------------------------------------
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
                "eta": eta,
                "k": k,
                "seed": seed,
                "generate_returncode": 0,
                "generated_eval_returncode": 0,
                "generate_log": str(gen_log),
                "generated_eval_log": str(eval_log),
                "graph_path": str(graph_path),
                "summary_csv": str(summary_csv),
                "selected_auc_key": auc_key,
                "selected_auc": auc_val,
                "selected_sp_key": sp_key,
                "selected_sp": sp_val,
            }
            row.update(summary_row)
            per_run_rows.append(row)

            # incremental save
            write_csv(per_run_rows, out_dir / "per_run_results.csv")

    # ---------------------------------------------------------
    # aggregate across seeds
    # ---------------------------------------------------------
    agg_rows = aggregate_seed_rows(per_run_rows)
    write_csv(agg_rows, out_dir / "aggregated_results.csv")

    # choose keys for pareto
    # we aggregate selected_auc / selected_sp across seeds
    x_key = "selected_sp_mean"
    y_key = "selected_auc_mean"
    xerr_key = "selected_sp_std"
    yerr_key = "selected_auc_std"

    front_rows = pareto_front(agg_rows, x_key=x_key, y_key=y_key)
    write_csv(front_rows, out_dir / "pareto_front.csv")

    plot_pareto(
        agg_rows=agg_rows,
        front_rows=front_rows,
        x_key=x_key,
        y_key=y_key,
        xerr_key=xerr_key,
        yerr_key=yerr_key,
        title=args.plot_title,
        png_path=out_dir / "pareto_auc_vs_sp.png",
        pdf_path=out_dir / "pareto_auc_vs_sp.pdf",
    )

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_dir": str(repo_dir),
                "run_name": args.run_name,
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "num_samples": args.num_samples,
                "eta_values": args.eta_values,
                "k_values": args.k_values,
                "seeds": args.seeds,
                "pair_mode": args.pair_mode,
                "include_baseline": args.include_baseline,
                "baseline_k": args.baseline_k,
                "graph_variant": args.graph_variant,
                "auc_candidates": args.auc_candidates,
                "sp_candidates": args.sp_candidates,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
