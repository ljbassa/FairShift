#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize FairWire Stage-2 controller grid runs.")
    parser.add_argument(
        "--controller_root",
        type=Path,
        default=Path("wandb/cora/Sync/controller"),
    )
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--sort_by", default="generated/aggregate_lp/score_sp_abs_gap")
    parser.add_argument("--descending", action="store_true")
    return parser.parse_args()


def read_jsonl_last(path):
    if not path.exists():
        return {}
    last = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                last = json.loads(line)
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
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
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


def maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


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
    rows = []

    for run_dir in sorted(root.glob(f"{args.prefix}*")):
        if not run_dir.is_dir():
            continue
        metrics = read_jsonl_last(run_dir / "controller_metrics.jsonl")
        ckpt_args = load_args_from_checkpoint(run_dir)
        lp_summary_path = find_lp_summary(run_dir)
        lp_summary = read_csv_first(lp_summary_path) if lp_summary_path else {}

        row = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "last_epoch": metrics.get("epoch"),
            "loss": metrics.get("loss"),
            "fair_score_k": ckpt_args.get("fair_score_k"),
            "fair_score_eta": ckpt_args.get("fair_score_eta"),
            "fair_score_learn_k": ckpt_args.get("fair_score_learn_k"),
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
            "grad/k_nonzero",
            "grad/k_mean_abs",
            "grad/k_max_abs",
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
            "aggregate_lp/auc",
            "aggregate_lp/score_sp_abs_gap",
            "aggregate_lp/sp_abs_gap",
            "aggregate_value/linkpred_auc",
            "aggregate_fair_abs_gap",
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
            math.inf if not isinstance(row.get("loss"), (int, float)) else row.get("loss"),
        ),
        reverse=args.descending,
    )

    out_csv = args.out_csv or (root / f"{args.prefix}_summary.csv")
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
        "fair_score_eta",
        "controller_lr",
        "fair_weight",
        "utility_weight",
        "generated/aggregate_lp/auc",
        "generated/aggregate_lp/score_sp_abs_gap",
        "fair_controller_delta_final_abs_mean",
        "fair_controller_eta_min",
        "fair_controller_eta_max",
        "weighted/fair",
        "weighted/utility",
    ]
    for row in rows[:10]:
        print({key: row.get(key) for key in preview_keys if key in row})


if __name__ == "__main__":
    main()
