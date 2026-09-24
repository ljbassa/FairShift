#!/usr/bin/env python3
"""Run eta/k ablations with the settings of a trusted local controller run."""

import argparse
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
VARIANTS = {
    "shared_eta": ("shared", "per_step"),
    "k_one": ("per_step", "fixed_one"),
    "shared_eta_k_one": ("shared", "fixed_one"),
    "baseline": ("per_step", "per_step"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_run", type=Path, required=True,
                        help="Existing run directory containing trusted args.pickle")
    parser.add_argument("--output_root", type=Path, required=True,
                        help="New runs go under OUTPUT_ROOT/{sp,eo}/VARIANT_seedN")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["shared_eta", "k_one", "shared_eta_k_one"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Default: the saved baseline training seed")
    parser.add_argument("--device", default=None, help="Default: the saved baseline device")
    parser.add_argument("--generation_seed", type=int, default=None,
                        help="Optional RNG reset before graph export; otherwise retain baseline behavior")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print validated commands without writing outputs or training")
    parser.add_argument("--run_generated_eval", action="store_true",
                        help="Also evaluate each run's exported graphs with the existing GAE evaluator")
    return parser.parse_args()


def training_parser():
    # Use the trainer's actual parser, including boolean actions and list arguments.
    # Import lazily so launcher --help does not require the training dependencies.
    sys.path.insert(0, str(REPO))
    from train_controller import build_parser
    return build_parser()


def serialize_arguments(parser, values):
    command = []
    for action in parser._actions:
        if action.dest == "help" or action.dest not in values:
            continue
        value = values[action.dest]
        if value is None:
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                command.append(option)
        elif isinstance(action, argparse._StoreAction):
            items = value if isinstance(value, (list, tuple)) else [value]
            command.extend([option, *[str(item) for item in items]])
        else:
            raise ValueError(f"Unsupported trainer argument action: {option}")
    # Validate choices/types using precisely the parser the subprocess will use.
    parsed = vars(parser.parse_args(command))
    return command, parsed


def build_plan(args, baseline, parser):
    output_root = args.output_root.resolve()
    # Match train_controller.make_log_dir's treatment of metric directories.
    if output_root.name in {"sp", "eo"}:
        output_root = output_root.parent
    seeds = args.seeds if args.seeds is not None else [int(baseline.get("seed", 0))]
    if len(set(seeds)) != len(seeds) or len(set(args.variants)) != len(args.variants):
        raise ValueError("Seeds and variants must not contain duplicates")
    for key in ("fair_score_eta_mode", "fair_score_k_mode"):
        if baseline.get(key, "per_step") != "per_step":
            raise ValueError(f"Baseline must use per_step eta and k; saved {key}={baseline[key]}")
    checkpoint = baseline.get("controller_pretrained_ckpt")
    if not checkpoint or not baseline.get("dataset"):
        raise ValueError("Baseline args must include controller_pretrained_ckpt and dataset")
    checkpoint = Path(checkpoint)
    checkpoint = checkpoint if checkpoint.is_absolute() else REPO / checkpoint
    if not 0.0 < float(baseline.get("fair_score_k", 0.15)) < 1.0:
        raise ValueError("The baseline fair_score_k must lie strictly between 0 and 1")

    plans = []
    for variant in args.variants:
        eta_mode, k_mode = VARIANTS[variant]
        for seed in seeds:
            values = dict(baseline)
            # Saved args are written AFTER prepare_data_args. Undo its mutation
            # so the original no-node-class branch and derived attrs are restored.
            if baseline.get("has_node_feature") is False:
                values["final_prob_node"] = None
            values.update(name=f"{variant}_seed{seed}", seed=seed,
                          controller_root=str(output_root), log_home=str(output_root),
                          controller_pretrained_ckpt=str(checkpoint.resolve()),
                          fair_score_eta_mode=eta_mode, fair_score_k_mode=k_mode)
            if args.device is not None:
                values["device"] = args.device
            if args.generation_seed is not None:
                values["generation_seed"] = args.generation_seed
            cli_args, settings = serialize_arguments(parser, values)
            run_dir = output_root / settings["fair_score_metric"] / settings["name"]
            graph_path = run_dir / "generated_samples/controller_best.pyg_full.pt"
            # Existing grid evaluations use seed 0, independent of controller seed.
            eval_command = [sys.executable, str(REPO / "evaluate_generated_graphs.py"),
                            "--graph_path", str(graph_path), "--dataset", settings["dataset"],
                            "--label_attr", settings["fair_label_attr"],
                            "--sensitive_attr", settings["fair_label_attr"],
                            "--device", settings["device"], "--seed", "0"]
            plans.append({
                "variant": variant, "seed": seed, "baseline_run": str(args.baseline_run.resolve()),
                "run_dir": str(run_dir), "settings": settings,
                "command": [sys.executable, str(REPO / "train_controller.py"), *cli_args],
                "eval_command": eval_command if args.run_generated_eval else None,
                "graph_path": str(graph_path), "cwd": str(REPO),
                "restored_final_prob_node_none": baseline.get("has_node_feature") is False,
            })
    return plans


def validate_sources(plans):
    # Check all inputs and destinations before starting any of the ablations.
    for plan in plans:
        settings = plan["settings"]
        checkpoint = Path(settings["controller_pretrained_ckpt"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint is missing: {checkpoint}")
        graphs = [REPO / "graphs" / f"{settings['dataset']}{suffix}.pkl"
                  for suffix in ("_feat", "")]
        if not any(path.is_file() for path in graphs):
            raise FileNotFoundError(f"Dataset graph is missing: {graphs[0]} (or {graphs[1]})")
        if Path(plan["run_dir"]).exists():
            raise FileExistsError(f"Refusing to overwrite an existing run: {plan['run_dir']}")


def execute_plan(plan):
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = run_dir / "ablation_manifest.json"

    def save(status):
        plan["status"] = status
        manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    plan["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save("training")
    try:
        with (run_dir / "train.log").open("x", encoding="utf-8") as log:
            subprocess.run(plan["command"], cwd=REPO, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        if not Path(plan["graph_path"]).is_file():
            raise FileNotFoundError(f"Training did not export graphs: {plan['graph_path']}")
        if plan["eval_command"]:
            save("evaluating")
            with (run_dir / "eval.log").open("x", encoding="utf-8") as log:
                subprocess.run(plan["eval_command"], cwd=REPO, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
    except (OSError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        plan["error"] = str(error)
        save("failed")
        raise
    plan["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save("complete")


def main():
    args = parse_args()
    source = args.baseline_run.resolve() / "args.pickle"
    with source.open("rb") as handle:
        saved = pickle.load(handle)
    baseline = saved if isinstance(saved, dict) else vars(saved)
    plans = build_plan(args, baseline, training_parser())
    validate_sources(plans)
    print(f"Baseline: {args.baseline_run.resolve()}", flush=True)
    print("Shared eta is one learned parameter reused at every diffusion step.", flush=True)
    if float(baseline.get("fair_score_k_tracking_loss_weight", 1.0)) == 0:
        print("Baseline tracking weight is 0: per_step k stays at its initial value; "
              "the other loss branches detach k. This setting is preserved.", flush=True)
    for plan in plans:
        print(f"\n[{plan['variant']}, seed {plan['seed']}] {plan['run_dir']}", flush=True)
        print(shlex.join(plan["command"]), flush=True)
        if plan["eval_command"]:
            print(shlex.join(plan["eval_command"]), flush=True)
        if not args.dry_run:
            execute_plan(plan)
            print(f"Completed: {plan['run_dir']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"[ablations] {error}", file=sys.stderr)
        sys.exit(1)
