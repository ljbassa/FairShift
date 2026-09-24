#!/usr/bin/env python3
"""Measure FairShift graph generation through completed AUC/SP evaluation.

Use the existing trained FairWire checkpoint and the evaluator's complete
sample.py-compatible GAE protocol. Select a physical GPU externally with
CUDA_VISIBLE_DEVICES; both children use logical cuda:0.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=["cora", "citeseer", "amazon_photo"])
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--eta-schedule", choices=["constant", "early", "late"], default="constant")
    parser.add_argument("--shift-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--python", default=sys.executable, help="Python with FairWire dependencies installed")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New directory for graph, logs, metrics, and timing reports")
    args = parser.parse_args()
    if not math.isfinite(args.eta) or args.eta <= 0:
        parser.error("--eta must be finite and positive; eta=0 disables FairShift")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.shift_clip is not None and (not math.isfinite(args.shift_clip) or args.shift_clip <= 0):
        parser.error("--shift-clip must be finite and positive")
    return args


def locate_reference(graph_path, dataset):
    """Use the evaluator's reference search order, without importing torch."""
    candidates = []
    for root in [graph_path.parent, *graph_path.parent.parents, REPO_ROOT, *REPO_ROOT.parents]:
        for name in [f"{dataset}_feat.pkl", f"{dataset}.pkl"]:
            candidates.extend([root / "graphs" / name, root / name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Missing {dataset} reference graph. Prepare {REPO_ROOT / 'graphs' / (dataset + '_feat.pkl')} "
        "with make_reference_graph.py before measuring this pipeline."
    )


def read_csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_metrics(per_graph_path, summary_path, num_samples):
    rows = read_csv_rows(per_graph_path)
    if len(rows) != num_samples:
        raise RuntimeError(f"Expected {num_samples} evaluated graphs, found {len(rows)}")
    metrics = []
    for index, row in enumerate(rows):
        if row.get("lp/error"):
            raise RuntimeError(f"GAE evaluation failed for graph {index}: {row['lp/error']}")
        auc = float(row["lp/auc"])
        sp = float(row["lp/sp_abs_gap"])
        if not all(math.isfinite(value) for value in (auc, sp)):
            raise RuntimeError(f"Non-finite AUC/SP for graph {index}: AUC={auc}, SP={sp}")
        if not (0 <= auc <= 1 and 0 <= sp <= 1):
            raise RuntimeError(f"AUC/SP outside [0, 1] for graph {index}: AUC={auc}, SP={sp}")
        if row.get("lp_protocol") != "samplepy_gae":
            raise RuntimeError(f"Unexpected evaluation protocol: {row.get('lp_protocol')}")
        metrics.append({"graph_index": index, "auc": auc, "sp_abs_gap": sp})
    summaries = read_csv_rows(summary_path)
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one summary row, found {len(summaries)}")
    summary = summaries[0]
    if int(float(summary["num_evaluated_graphs"])) != num_samples:
        raise RuntimeError("Summary graph count differs from requested count")
    auc_mean = float(summary["lp/auc_mean"])
    sp_mean = float(summary["lp/sp_abs_gap_mean"])
    if not all(math.isfinite(value) for value in (auc_mean, sp_mean)):
        raise RuntimeError("Summary AUC/SP are not finite")
    return metrics, auc_mean, sp_mean


def save_report(report, output_dir):
    json_path = output_dir / "timing.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    row = {key: report.get(key) for key in [
        "status", "dataset", "num_samples", "eta", "eta_schedule", "seed",
        "generation_seconds", "evaluation_seconds", "end_to_end_seconds",
        "auc_mean", "sp_abs_gap_mean", "checkpoint", "reference_graph",
        "generated_graph", "error",
    ]}
    with (output_dir / "timing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained checkpoint does not exist: {checkpoint}")
    python_exec = shutil.which(args.python)
    if python_exec is None:
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = Path(__file__).parent / f"pipeline_{args.dataset}_{timestamp}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; use a new directory: {output_dir}")
    graph_path = output_dir / "generated.pyg.pt"
    per_graph_path = output_dir / "evaluation.per_graph.csv"
    summary_path = output_dir / "evaluation.summary.csv"
    reference = locate_reference(graph_path, args.dataset)
    sample_script = REPO_ROOT / "sample.py"
    eval_script = REPO_ROOT / "evaluate_generated_graphs.py"
    for script in [sample_script, eval_script]:
        if not script.is_file():
            raise FileNotFoundError(script)

    sample_cmd = [
        python_exec, "-u", str(sample_script),
        "--model_path", str(checkpoint),
        "--num_samples", str(args.num_samples),
        "--gpu", "0", "--seed", str(args.seed),
        "--sp_shift", "--sp_eta", str(args.eta),
        "--sp_eta_schedule", args.eta_schedule,
        "--save_pt_path", str(graph_path), "--skip_internal_eval",
    ]
    if args.shift_clip is not None:
        sample_cmd.extend(["--sp_shift_clip", str(args.shift_clip)])
    eval_cmd = [
        python_exec, "-u", str(eval_script),
        "--graph_path", str(graph_path), "--dataset", args.dataset,
        "--seed", str(args.seed), "--sensitive_attr", "sens",
        "--out_per_graph_csv", str(per_graph_path),
        "--out_summary_csv", str(summary_path),
    ]
    report = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "eta": args.eta,
        "eta_schedule": args.eta_schedule,
        "shift_clip": args.shift_clip,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "reference_graph": str(reference),
        "generated_graph": str(graph_path),
        "per_graph_metrics": str(per_graph_path),
        "summary_metrics": str(summary_path),
        "hostname": platform.node(),
        "python": python_exec,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "child_environment_overrides": {"PYTHONDONTWRITEBYTECODE": "1"},
        "generation_command": sample_cmd,
        "evaluation_command": eval_cmd,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [sample_script, eval_script, REPO_ROOT / "Model" / "fair_diffusion.py"]
        },
        "timing_scope": {
            "start": "Immediately before starting the graph-generation subprocess",
            "end": "After evaluation subprocess completion and validation of finite AUC/SP CSV output",
            "includes": [
                "Child Python startup and imports, checkpoint/dataset loading and preprocessing",
                "One full FairShift-controlled sampling run for the requested number of graphs",
                "Saving/loading generated graphs and loading the prepared reference graph",
                "Full sample.py-compatible GAE hyperparameter search and training",
                "AUC/SP computation, metric CSV writing, child-process shutdown and result validation",
            ],
            "excludes": [
                "Training the FairWire backbone checkpoint",
                "One-time preparation of the reference graph",
                "Runner preflight, timing report serialization and user-facing output after validation",
            ],
            "evaluator_protocol": {
                "name": "samplepy_gae",
                "configurations_max": 48,
                "lr": [0.03, 0.01, 0.003, 0.001],
                "hidden_size": [16, 32, 128, 512],
                "dropout": [0.0, 0.1, 0.2],
                "num_layers": 1,
                "epochs_max_per_configuration": 1000,
                "early_stop_patience": 5,
                "grid_early_exit": "Stop the grid if a trial reaches validation AUC 1.0",
                "split": "80/10/10 positive edges, equal negative val/test pairs",
                "sp": "Absolute mean predicted probability gap: same-sensitive vs different-sensitive test pairs",
            },
            "synchronization": "Sequential subprocess completion; each child exits before its wall timer stops",
        },
    }
    output_dir.mkdir(parents=True)
    save_report(report, output_dir)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Generation: {shlex.join(sample_cmd)}", flush=True)
    print(f"Evaluation: {shlex.join(eval_cmd)}", flush=True)

    def run_stage(name, command):
        log_path = output_dir / f"{name}.log"
        stage_start = time.perf_counter()
        with log_path.open("w") as log:
            completed = subprocess.run(
                command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        report[f"{name}_seconds"] = time.perf_counter() - stage_start
        report[f"{name}_returncode"] = completed.returncode
        report[f"{name}_log"] = str(log_path)
        if completed.returncode != 0:
            raise RuntimeError(f"{name} failed with exit code {completed.returncode}; see {log_path}")
        print(f"{name}: {report[f'{name}_seconds']:.3f} seconds", flush=True)

    total_start = time.perf_counter()
    try:
        run_stage("generation", sample_cmd)
        if not graph_path.is_file():
            raise RuntimeError("Generation exited successfully without producing the graph file")
        expected_dataset_message = f" model trained on {args.dataset}\n"
        if expected_dataset_message not in Path(report["generation_log"]).read_text():
            raise RuntimeError(
                "Checkpoint dataset does not match --dataset, or the expected sampler dataset message is missing"
            )
        run_stage("evaluation", eval_cmd)
        metrics, auc_mean, sp_mean = validate_metrics(per_graph_path, summary_path, args.num_samples)
        report["end_to_end_seconds"] = time.perf_counter() - total_start
        report.update({
            "status": "complete", "graphs": metrics,
            "auc_mean": auc_mean, "sp_abs_gap_mean": sp_mean,
        })
    except Exception as exc:
        report["end_to_end_seconds"] = time.perf_counter() - total_start
        report["status"] = "failed"
        report["error"] = str(exc)
        save_report(report, output_dir)
        raise
    save_report(report, output_dir)
    print(
        f"{args.dataset}: {report['end_to_end_seconds']:.3f} seconds end to end; "
        f"AUC={auc_mean:.6f}; SP={sp_mean:.6f}\n"
        f"Timing reports: {output_dir / 'timing.json'} and {output_dir / 'timing.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
