#!/usr/bin/env python3
import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Stage-2 controller grid runs.")
    parser.add_argument(
        "--controller_root",
        type=Path,
        default=Path("wandb/cora/multinomial_diffusion/controller"),
    )
    parser.add_argument("--fair_score_metric", choices=["sp", "eo"], default="sp")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--sort_by", default=None)
    parser.add_argument("--descending", action="store_true")
    args = parser.parse_args()
    if args.sort_by is None:
        args.sort_by = "lp/eo_abs_gap_mean" if args.fair_score_metric == "eo" else "eval/value/fair_edge_sp_abs_gap"
    return args


def read_jsonl_last(path):
    """Return the last metrics object, warning about damaged log records."""
    if not path.exists():
        return {}
    last = {}
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"{path}:{line_number}: skipping invalid JSON record ({exc})",
                    RuntimeWarning,
                )
                continue
            if not isinstance(record, dict):
                warnings.warn(
                    f"{path}:{line_number}: skipping non-object JSON record "
                    f"({type(record).__name__})",
                    RuntimeWarning,
                )
                continue
            last = record
    return last


def read_csv_first(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        return {}
    out = {}
    for key, value in rows[0].items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def load_args_from_checkpoint(run_dir):
    path = run_dir / "check" / "controller_last.pt"
    if not path.exists():
        path = run_dir / "check" / "controller_best.pt"
    if not path.exists():
        return {}
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    return ckpt.get("args", {}) or {}


def find_lp_summary(run_dir):
    sample_dir = run_dir / "generated_samples"
    if not sample_dir.exists():
        return None
    candidates = sorted(sample_dir.glob("*summary.csv"))
    return candidates[-1] if candidates else None


def maybe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def weighted_terms(row, args):
    fair_w = float(args.get("fair_score_fair_loss_weight", math.nan))
    k_w = float(args.get("fair_score_k_tracking_loss_weight", math.nan))
    util_w = float(args.get("fair_score_utility_loss_weight", math.nan))
    out = {}
    if "fair_controller_fair_loss" in row and math.isfinite(fair_w):
        out["weighted/fair"] = float(row["fair_controller_fair_loss"]) * fair_w
    if "fair_controller_k_tracking_loss" in row and math.isfinite(k_w):
        out["weighted/k_tracking"] = float(row["fair_controller_k_tracking_loss"]) * k_w
    if "fair_controller_utility_loss" in row and math.isfinite(util_w):
        out["weighted/utility"] = float(row["fair_controller_utility_loss"]) * util_w
    return out


def main():
    args = parse_args()
    root = args.controller_root
    if root.name in {"sp", "eo"}:
        root = root.parent / args.fair_score_metric
    elif args.fair_score_metric == "eo" or (root / args.fair_score_metric).is_dir():
        root = root / args.fair_score_metric
    rows = []

    for run_dir in sorted(root.glob(f"{args.prefix}*")):
        if not run_dir.is_dir():
            continue
        metrics = read_jsonl_last(run_dir / "controller_metrics.jsonl")
        ckpt_args = load_args_from_checkpoint(run_dir)
        run_metric = ckpt_args.get("fair_score_metric", metrics.get("fair_score_metric", "sp"))
        if run_metric != args.fair_score_metric:
            continue
        lp_summary_path = find_lp_summary(run_dir)
        lp_summary = read_csv_first(lp_summary_path) if lp_summary_path else {}

        row = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "last_epoch": metrics.get("epoch"),
            "loss": metrics.get("loss"),
            "fair_score_k": ckpt_args.get("fair_score_k"),
            "fair_score_eta": ckpt_args.get("fair_score_eta"),
            "fair_score_metric": run_metric,
            "controller_lr": ckpt_args.get("controller_lr"),
            "fair_weight": ckpt_args.get("fair_score_fair_loss_weight"),
            "k_tracking_weight": ckpt_args.get("fair_score_k_tracking_loss_weight"),
            "utility_weight": ckpt_args.get("fair_score_utility_loss_weight"),
            "replay_num_samples": ckpt_args.get("controller_replay_num_samples"),
            "replay_refresh": ckpt_args.get("controller_replay_refresh"),
        }

        keep_metric_keys = [
            "fair_controller_fair_loss",
            "fair_controller_k_tracking_loss",
            "fair_controller_utility_loss",
            "fair_controller_delta_final_abs_mean",
            "fair_controller_gap_final_abs_mean",
            "fair_controller_valid_graphs",
            "fair_controller_total_graphs",
            "fair_controller_eo_positive_mass_same_min",
            "fair_controller_eo_positive_mass_same_mean",
            "fair_controller_eo_positive_mass_diff_min",
            "fair_controller_eo_positive_mass_diff_mean",
            "fair_controller_mean_abs_shift",
            "fair_guidance_raw_abs_mean",
            "fair_guidance_shift_abs_mean",
            "fair_guidance_delta_pre_abs_mean",
            "fair_controller_eta_mean",
            "fair_controller_eta_min",
            "fair_controller_eta_max",
            "fair_controller_k_mean",
            "fair_controller_k_min",
            "fair_controller_k_max",
            "grad/eta_nonzero",
            "grad/eta_mean_abs",
            "grad/eta_max_abs",
            "eval/value/linkpred_auc",
            "eval/value/fair_edge_sp_abs_gap",
            "eval/nmae/d",
            "eval/nmae/triangle_count",
            "eval/nmae/clustering_coefficient",
            "eval/fair_guidance_mean_abs_shift",
        ]
        for key in keep_metric_keys:
            if key in metrics:
                row[key] = maybe_float(metrics[key])
        row.update(weighted_terms(row, ckpt_args))

        lp_keep = [
            "lp/auc_mean",
            "lp/auc_std",
            "lp/score_sp_gap_mean",
            "lp/score_sp_gap_std",
            "lp/score_sp_abs_gap_mean",
            "lp/score_sp_abs_gap_std",
            "lp/sp_abs_gap_mean",
            "lp/eo_gap_mean",
            "lp/eo_gap_std",
            "lp/eo_abs_gap_mean",
            "lp/eo_abs_gap_std",
            "lp/eo_defined_mean",
            "lp/eo_num_pos_sensitive_mean",
            "lp/eo_num_pos_nonsensitive_mean",
            "aggregate_lp/auc",
            "aggregate_lp/score_sp_abs_gap",
            "aggregate_lp/sp_abs_gap",
            "aggregate_lp/eo_gap",
            "aggregate_lp/eo_abs_gap",
            "aggregate_value/linkpred_auc",
        ]
        for key in lp_keep:
            if key in lp_summary:
                row[key] = lp_summary[key]
                row[f"generated/{key}"] = lp_summary[key]

        rows.append(row)

    if not rows:
        raise SystemExit(f"No runs found under {root} with prefix {args.prefix!r}")

    sort_key = args.sort_by
    rows.sort(
        key=lambda row: (
            math.inf if not isinstance(row.get(sort_key), (int, float)) else row.get(sort_key),
            math.inf if not isinstance(row.get("loss"), (int, float)) else row["loss"],
        ),
        reverse=args.descending,
    )

    out_csv = args.out_csv or (root / f"{args.prefix}_summary.csv")
    out_base = out_csv.parent.parent if out_csv.parent.name in {"sp", "eo"} else out_csv.parent
    out_csv = out_base / args.fair_score_metric / out_csv.name
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote summary: {out_csv}")
    print("top rows:")
    preview_keys = [
        "run_name",
        "loss",
        "fair_score_k",
        "eval/value/linkpred_auc",
    ]
    if args.fair_score_metric == "sp":
        preview_keys.append("eval/value/fair_edge_sp_abs_gap")
    preview_keys += [
        "lp/auc_mean",
        "lp/auc_std",
    ]
    if args.fair_score_metric == "eo":
        preview_keys += [
            "lp/eo_abs_gap_mean",
            "lp/eo_abs_gap_std",
            "lp/eo_defined_mean",
            "generated/aggregate_lp/auc",
            "generated/aggregate_lp/eo_abs_gap",
        ]
    else:
        preview_keys += [
            "lp/score_sp_gap_mean",
            "lp/score_sp_gap_std",
            "generated/aggregate_lp/auc",
            "generated/aggregate_lp/score_sp_abs_gap",
        ]
    preview_keys += [
        "fair_controller_eta_min",
        "fair_controller_eta_max",
        "fair_controller_mean_abs_shift",
    ]
    for row in rows[:10]:
        print({key: row.get(key) for key in preview_keys if key in row})


if __name__ == "__main__":
    main()
