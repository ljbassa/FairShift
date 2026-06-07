#!/usr/bin/env python3
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {value}")


def tag_float(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a FairWire Stage-2 fairness-controller grid with fixed-k controller training."
    )
    parser.add_argument("--repo_dir", type=Path, default=Path.cwd())
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name used for the controller root and Pareto title. Defaults to inferring from --stage1_ckpt.",
    )
    parser.add_argument("--name_prefix", default="cora_T8_kfixed_controller_grid")
    parser.add_argument("--controller_root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)

    # Match EDGE_fairness_loss_after argument names.
    parser.add_argument("--eta_values", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--controller_lrs", type=float, nargs="+", default=[5e-4, 1e-3])
    parser.add_argument("--fair_weights", type=float, nargs="+", default=[5e4, 1e5])
    parser.add_argument("--utility_weights", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--k_tracking_weights", type=float, nargs="+", default=[0.0])
    parser.add_argument("--fair_score_k", type=float, default=0.15)
    parser.add_argument(
        "--fair_score_k_values",
        type=float,
        nargs="+",
        default=None,
        help="Grid over fixed k values. Defaults to the single --fair_score_k value.",
    )

    parser.add_argument("--controller_epochs", type=int, default=1000)
    parser.add_argument("--controller_replay_num_samples", type=int, default=1)
    parser.add_argument("--controller_replay_refresh", type=int, default=100)
    parser.add_argument("--sample_batch_size", type=int, default=32768)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_generation", type=int, default=64)
    parser.add_argument("--fair_label_attr", default="y")
    parser.add_argument("--fair_score_guidance_normalize", type=str2bool, default=True)

    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_home", default="./wandb")
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--check_every", type=int, default=None)
    parser.add_argument("--clip_norm", type=float, default=None)
    parser.add_argument("--clip_value", type=float, default=None)

    parser.add_argument("--run_generated_eval", action="store_true")
    parser.add_argument("--generated_eval_max_graphs", type=int, default=None)
    parser.add_argument("--generated_eval_lp_epochs", type=int, default=None)
    parser.add_argument("--generated_eval_lp_patience", type=int, default=None)
    parser.add_argument("--generated_eval_lp_batch_size", type=int, default=None)
    parser.add_argument(
        "--skip_generated_pareto",
        action="store_true",
        help="when --run_generated_eval is set, skip the final LP Pareto summary/plot",
    )
    parser.add_argument("--pareto_summary_csv", type=Path, default=None)
    parser.add_argument("--pareto_plot_path", type=Path, default=None)
    parser.add_argument("--pareto_x_metric", default="lp/score_sp_abs_gap_mean")
    parser.add_argument("--pareto_y_metric", default="lp/auc_mean")
    parser.add_argument("--pareto_label_points", choices=["none", "front", "all"], default="front")
    parser.add_argument("--pareto_title", default=None)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra train_controller.py args after --",
    )
    args = parser.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    if args.fair_score_k_values is None:
        args.fair_score_k_values = [args.fair_score_k]
    return args


def infer_dataset_from_stage1_ckpt(stage1_ckpt):
    ckpt_path = Path(stage1_ckpt)
    parent = ckpt_path.parent.name
    if parent.endswith("_cpts"):
        parts = parent[:-5].split("_")
        if len(parts) >= 3:
            return "_".join(parts[:-2])
    return "cora"


def make_run_name(prefix, eta, lr, fair_w, util_w, k_w, fair_score_k, normalize):
    norm_tag = "norm" if normalize else "raw"
    return (
        f"{prefix}_{norm_tag}"
        f"_kfixed{tag_float(fair_score_k)}"
        f"_eta{tag_float(eta)}"
        f"_lr{tag_float(lr)}"
        f"_fw{tag_float(fair_w)}"
        f"_uw{tag_float(util_w)}"
        f"_kw{tag_float(k_w)}"
    )


def build_command(args, run_name, eta, lr, fair_w, util_w, k_w, fair_score_k):
    cmd = [
        args.python_exec,
        "train_controller.py",
        "--name", run_name,
        "--controller_pretrained_ckpt", args.stage1_ckpt,
        "--log_home", args.log_home,
        "--device", args.device,
        "--seed", str(args.seed),
        "--controller_epochs", str(args.controller_epochs),
        "--controller_lr", str(lr),
        "--controller_replay_num_samples", str(args.controller_replay_num_samples),
        "--controller_replay_refresh", str(args.controller_replay_refresh),
        "--sample_batch_size", str(args.sample_batch_size),
        "--num_workers", str(args.num_workers),
        "--num_generation", str(args.num_generation),
        "--fair_label_attr", args.fair_label_attr,
        "--fair_score_eta", str(eta),
        "--fair_score_k", str(fair_score_k),
        "--fair_score_learn_k", "False",
        "--fair_score_learn_eta", "True",
        "--fair_score_guidance_normalize", str(bool(args.fair_score_guidance_normalize)),
        "--fair_score_train_loss_weight", "1.0",
        "--fair_score_fair_loss_weight", str(fair_w),
        "--fair_score_k_tracking_loss_weight", str(k_w),
        "--fair_score_utility_loss_weight", str(util_w),
    ]
    if args.eval_every is not None:
        cmd += ["--eval_every", str(args.eval_every)]
    if args.check_every is not None:
        cmd += ["--check_every", str(args.check_every)]
    if args.clip_norm is not None:
        cmd += ["--clip_norm", str(args.clip_norm)]
    if args.clip_value is not None:
        cmd += ["--clip_value", str(args.clip_value)]
    if args.run_generated_eval:
        cmd.append("--run_generated_eval")
    if args.generated_eval_max_graphs is not None:
        cmd += ["--generated_eval_max_graphs", str(args.generated_eval_max_graphs)]
    if args.generated_eval_lp_epochs is not None:
        cmd += ["--generated_eval_lp_epochs", str(args.generated_eval_lp_epochs)]
    if args.generated_eval_lp_patience is not None:
        cmd += ["--generated_eval_lp_patience", str(args.generated_eval_lp_patience)]
    if args.generated_eval_lp_batch_size is not None:
        cmd += ["--generated_eval_lp_batch_size", str(args.generated_eval_lp_batch_size)]
    cmd += args.extra_args
    return cmd


def resolve_path(path, repo_dir):
    if path is None:
        return None
    if path.is_absolute():
        return path
    return repo_dir / path


def run_generated_pareto(args, repo_dir, controller_root, dataset):
    summary_csv = resolve_path(args.pareto_summary_csv, repo_dir)
    if summary_csv is None:
        summary_csv = controller_root / f"{args.name_prefix}_summary.csv"
    plot_path = resolve_path(args.pareto_plot_path, repo_dir)
    if plot_path is None:
        plot_path = controller_root / f"{args.name_prefix}_pareto_lp_auc_vs_score_sp.jpg"

    summarize_script = repo_dir / "scripts" / "summarize_controller_grid.py"
    plot_script = repo_dir / "scripts" / "plot_controller_grid_pareto.py"

    summary_cmd = [
        args.python_exec,
        str(summarize_script),
        "--controller_root", str(controller_root),
        "--prefix", args.name_prefix,
        "--sort_by", args.pareto_x_metric,
        "--out_csv", str(summary_csv),
    ]
    plot_cmd = [
        args.python_exec,
        str(plot_script),
        "--summary_csv", str(summary_csv),
        "--out_path", str(plot_path),
        "--x_metric", args.pareto_x_metric,
        "--y_metric", args.pareto_y_metric,
        "--label_points", args.pareto_label_points,
        "--title", args.pareto_title or f"{dataset}: Controller LP Pareto",
    ]

    print("[grid] generated LP Pareto summary:")
    print(" ".join(summary_cmd))
    summary_proc = subprocess.run(summary_cmd, cwd=repo_dir)
    if summary_proc.returncode != 0:
        return summary_csv, plot_path, summary_proc.returncode

    print("[grid] generated LP Pareto plot:")
    print(" ".join(plot_cmd))
    plot_proc = subprocess.run(plot_cmd, cwd=repo_dir)
    return summary_csv, plot_path, plot_proc.returncode


def main():
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    dataset = args.dataset or infer_dataset_from_stage1_ckpt(args.stage1_ckpt)
    controller_root = args.controller_root
    if controller_root is None:
        controller_root = repo_dir / "wandb" / dataset / "Sync" / "controller"
    if not controller_root.is_absolute():
        controller_root = repo_dir / controller_root

    manifest = args.manifest
    if manifest is None:
        manifest = controller_root / f"{args.name_prefix}_manifest.jsonl"
    manifest = resolve_path(manifest, repo_dir)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(
        args.fair_score_k_values,
        args.eta_values,
        args.controller_lrs,
        args.fair_weights,
        args.utility_weights,
        args.k_tracking_weights,
    ))
    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    print(f"[grid] repo_dir={repo_dir}")
    print(f"[grid] dataset={dataset}")
    print(f"[grid] controller_root={controller_root}")
    print(f"[grid] manifest={manifest}")
    print(f"[grid] num_runs={len(combos)}")

    for run_idx, (fair_score_k, eta, lr, fair_w, util_w, k_w) in enumerate(combos, 1):
        run_name = make_run_name(
            args.name_prefix,
            eta=eta,
            lr=lr,
            fair_w=fair_w,
            util_w=util_w,
            k_w=k_w,
            fair_score_k=fair_score_k,
            normalize=args.fair_score_guidance_normalize,
        )
        final_ckpt = controller_root / run_name / "check" / "controller_final.pt"
        cmd = build_command(args, run_name, eta, lr, fair_w, util_w, k_w, fair_score_k)
        record = {
            "run_idx": run_idx,
            "num_runs": len(combos),
            "run_name": run_name,
            "fair_score_k": fair_score_k,
            "fair_score_eta": eta,
            "controller_lr": lr,
            "fair_weight": fair_w,
            "utility_weight": util_w,
            "k_tracking_weight": k_w,
            "learn_k": False,
            "command": cmd,
            "status": "pending",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(f"\n[grid] {run_idx}/{len(combos)} {run_name}")
        print(" ".join(cmd))

        if args.skip_existing and final_ckpt.exists():
            record["status"] = "skipped_existing"
            record["returncode"] = 0
            with manifest.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record) + "\n")
            print(f"[grid] skip existing: {final_ckpt}")
            continue

        if args.dry_run:
            record["status"] = "dry_run"
            record["returncode"] = 0
            with manifest.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record) + "\n")
            continue

        start = time.time()
        proc = subprocess.run(cmd, cwd=repo_dir)
        record["elapsed_sec"] = round(time.time() - start, 3)
        record["returncode"] = proc.returncode
        record["status"] = "ok" if proc.returncode == 0 else "failed"
        record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with manifest.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

        if proc.returncode != 0 and args.fail_fast:
            raise SystemExit(proc.returncode)

    if args.run_generated_eval and not args.skip_generated_pareto and not args.dry_run:
        summary_csv, plot_path, pareto_returncode = run_generated_pareto(args, repo_dir, controller_root, dataset)
        if pareto_returncode == 0:
            print(f"[grid] generated LP Pareto summary: {summary_csv}")
            print(f"[grid] generated LP Pareto plot: {plot_path}")
        else:
            print(f"[grid] generated LP Pareto failed with returncode={pareto_returncode}")
            if args.fail_fast:
                raise SystemExit(pareto_returncode)


if __name__ == "__main__":
    main()
