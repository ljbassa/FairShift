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
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {value}")


def tag_float(value):
    text = f"{value:g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Stage-2 fairness-controller grid.")
    parser.add_argument("--repo_dir", type=Path, default=Path.cwd())
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--name_prefix", default="stage2_cora_controller_grid")
    parser.add_argument("--controller_root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)

    parser.add_argument("--eta_values", type=float, nargs="+", default=[10000.0, 18000.0])
    parser.add_argument("--controller_lrs", type=float, nargs="+", default=[5e-4, 1e-3])
    parser.add_argument("--fair_weights", type=float, nargs="+", default=[1e5, 3e5])
    parser.add_argument("--utility_weights", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--k_tracking_weights", type=float, nargs="+", default=[0.01, 0.05])

    parser.add_argument("--controller_epochs", type=int, default=500)
    parser.add_argument("--controller_replay_num_samples", type=int, default=2)
    parser.add_argument("--controller_replay_refresh", type=int, default=10)
    parser.add_argument("--fair_score_k", type=float, default=0.5)
    parser.add_argument(
        "--fair_score_k_values",
        type=float,
        nargs="+",
        default=None,
        help="Optional grid over fixed initial k values. Defaults to the single --fair_score_k value.",
    )
    parser.add_argument("--fair_score_guidance_normalize", type=str2bool, default=False)

    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--diffusion_dim", type=int, default=128)
    parser.add_argument("--diffusion_steps", type=int, default=256)
    parser.add_argument("--noise_schedule", default="linear")
    parser.add_argument("--edge_dropout", type=float, default=0.05)
    parser.add_argument("--loss_type", default="vb_ce_xt_prescribred_st")
    parser.add_argument("--parametrization", default="xt_prescribed_st")
    parser.add_argument("--num_heads", nargs="+", default=["8", "8", "8", "8", "1"])

    parser.add_argument("--num_generation", type=int, default=8)
    parser.add_argument("--sample_batch_size", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--check_every", type=int, default=100)
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="adam")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument(
        "--run_generated_eval",
        action="store_true",
        help="after each successful controller run, run evaluate_generated_graphs.py on controller_best.pyg_full.pt",
    )
    parser.add_argument("--generated_eval_device", default=None)
    parser.add_argument("--generated_eval_label_attr", default="y")
    parser.add_argument("--generated_eval_sensitive_attr", default="y")
    parser.add_argument("--force_generated_eval", action="store_true")
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


def make_run_name(prefix, eta, lr, fair_w, util_w, k_w, fair_score_k, normalize):
    norm_tag = "norm" if normalize else "raw"
    return (
        f"{prefix}_{norm_tag}"
        f"_kinit{tag_float(fair_score_k)}"
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
        "--controller_epochs", str(args.controller_epochs),
        "--controller_lr", str(lr),
        "--controller_replay_num_samples", str(args.controller_replay_num_samples),
        "--controller_replay_refresh", str(args.controller_replay_refresh),
        "--diffusion_dim", str(args.diffusion_dim),
        "--diffusion_steps", str(args.diffusion_steps),
        "--noise_schedule", args.noise_schedule,
        "--edge_dropout", str(args.edge_dropout),
        "--device", args.device,
        "--dataset", args.dataset,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--optimizer", args.optimizer,
        "--final_prob_edge", "1", "0",
        "--sample_time_method", "importance",
        "--loss_type", args.loss_type,
        "--parametrization", args.parametrization,
        "--degree",
        "--num_heads", *[str(x) for x in args.num_heads],
        "--use_node_feat",
        "--fair_score_k", str(fair_score_k),
        "--fair_score_eta", str(eta),
        "--fair_score_fair_loss_weight", str(fair_w),
        "--fair_score_k_tracking_loss_weight", str(k_w),
        "--fair_score_utility_loss_weight", str(util_w),
        "--fair_score_guidance_normalize", str(bool(args.fair_score_guidance_normalize)),
        "--num_generation", str(args.num_generation),
        "--sample_batch_size", str(args.sample_batch_size),
        "--eval_every", str(args.eval_every),
        "--check_every", str(args.check_every),
    ]
    if args.clip_value is not None:
        cmd += ["--clip_value", str(args.clip_value)]
    cmd += args.extra_args
    return cmd


def run_generated_eval(args, repo_dir, controller_root, run_name):
    graph_path = controller_root / run_name / "generated_samples" / "controller_best.pyg_full.pt"
    summary_path = graph_path.with_name(f"{graph_path.name[:-3]}.overlap_lp_gae_summary.csv")
    per_graph_path = graph_path.with_name(f"{graph_path.name[:-3]}.overlap_lp_gae_per_graph.csv")

    if not graph_path.exists():
        print(f"[grid] generated graph not found; skipping LP eval: {graph_path}")
        return None, None
    if summary_path.exists() and not args.force_generated_eval:
        print(f"[grid] LP summary exists; skipping LP eval: {summary_path}")
        return summary_path, 0

    eval_device = args.generated_eval_device or args.device
    cmd = [
        args.python_exec,
        "evaluate_generated_graphs.py",
        "--graph_path", str(graph_path),
        "--dataset", args.dataset,
        "--label_attr", args.generated_eval_label_attr,
        "--sensitive_attr", args.generated_eval_sensitive_attr,
        "--device", eval_device,
        "--out_summary_csv", str(summary_path),
        "--out_per_graph_csv", str(per_graph_path),
    ]
    print("[grid] generated LP eval:")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo_dir)
    return summary_path, proc.returncode


def resolve_optional_path(path, repo_dir):
    if path is None:
        return None
    if path.is_absolute():
        return path
    return repo_dir / path


def run_generated_pareto(args, repo_dir, controller_root):
    summary_csv = resolve_optional_path(args.pareto_summary_csv, repo_dir)
    if summary_csv is None:
        summary_csv = controller_root / f"{args.name_prefix}_summary.csv"
    plot_path = resolve_optional_path(args.pareto_plot_path, repo_dir)
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
        "--title", f"{args.dataset}: Controller LP Pareto",
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
    controller_root = args.controller_root
    if controller_root is None:
        controller_root = repo_dir / "wandb" / args.dataset / "multinomial_diffusion" / "controller"
    if not controller_root.is_absolute():
        controller_root = repo_dir / controller_root

    manifest = args.manifest
    if manifest is None:
        manifest = controller_root / f"{args.name_prefix}_manifest.jsonl"
    if not manifest.is_absolute():
        manifest = repo_dir / manifest
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
            "eta": eta,
            "controller_lr": lr,
            "fair_weight": fair_w,
            "utility_weight": util_w,
            "k_tracking_weight": k_w,
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

        if proc.returncode == 0 and args.run_generated_eval:
            summary_path, eval_returncode = run_generated_eval(args, repo_dir, controller_root, run_name)
            record["generated_eval_summary"] = str(summary_path) if summary_path is not None else None
            record["generated_eval_returncode"] = eval_returncode
            if eval_returncode not in (None, 0):
                record["status"] = "generated_eval_failed"

        record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with manifest.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

        if args.fail_fast:
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)
            if record.get("generated_eval_returncode") not in (None, 0):
                raise SystemExit(record["generated_eval_returncode"])

    if args.run_generated_eval and not args.skip_generated_pareto and not args.dry_run:
        summary_csv, plot_path, pareto_returncode = run_generated_pareto(args, repo_dir, controller_root)
        if pareto_returncode == 0:
            print(f"[grid] generated LP Pareto summary: {summary_csv}")
            print(f"[grid] generated LP Pareto plot: {plot_path}")
        else:
            print(f"[grid] generated LP Pareto failed with returncode={pareto_returncode}")
            if args.fail_fast:
                raise SystemExit(pareto_returncode)


if __name__ == "__main__":
    main()
