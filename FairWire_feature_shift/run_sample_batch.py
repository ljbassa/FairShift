#!/usr/bin/env python3
import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DATASET_CHOICES = [
    "cora",
    "citeseer",
    "amazon_photo",
    "amazon_computer",
    "german",
    "pokec_n",
]

GPU_CHOICES = list(range(8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FairWire_feature/sample.py sequentially for multiple alphaA values."
    )
    parser.add_argument("--repo_dir", type=str, default=".", help="Path to the FairWire_feature repo root.")
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used to launch sample.py.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_CHOICES,
        help="Dataset name used to infer checkpoint directories.",
    )
    parser.add_argument(
        "-aX",
        "--alphaX",
        type=float,
        required=True,
        help="Feature fairness coefficient used in checkpoint directory names.",
    )
    parser.add_argument(
        "--alphaA_values",
        "--aA_values",
        type=float,
        nargs="+",
        required=True,
        help="One or more alphaA values to sample, for example: 0 0.1 1 5 10",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        choices=GPU_CHOICES,
        help="GPU id forwarded to sample.py.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of graphs generated per checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="batch_runs/sample",
        help="Directory where batch logs and summaries are stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed forwarded to sample.py.",
    )
    parser.add_argument(
        "--checkpoint_glob",
        type=str,
        default="Sync_T*.pth",
        help="Checkpoint glob searched inside each inferred checkpoint directory.",
    )
    parser.add_argument(
        "--save_pkl_root",
        type=str,
        default=None,
        help="If set, each alphaA run saves PKLs into a separate subdirectory under this root.",
    )
    parser.add_argument(
        "--save_pt_root",
        type=str,
        default=None,
        help="If set, each alphaA run saves a .pyg.pt file under this root.",
    )
    parser.add_argument(
        "--skip_internal_eval",
        action="store_true",
        help="Forward --skip_internal_eval to sample.py.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip alphaA values whose checkpoint is missing instead of stopping.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop the batch immediately when one sample.py run fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and write logs without actually launching sample.py.",
    )
    parser.add_argument(
        "extra_sample_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to sample.py. Put them after '--'.",
    )
    args = parser.parse_args()
    if args.extra_sample_args and args.extra_sample_args[0] == "--":
        args.extra_sample_args = args.extra_sample_args[1:]
    return args


def float_arg_text(value: float) -> str:
    return str(float(value))


def checkpoint_dir_name(dataset: str, alpha_a: float, alpha_x: float) -> str:
    return "{dataset}_{alpha_a}_{alpha_x}_cpts".format(
        dataset=dataset,
        alpha_a=float_arg_text(alpha_a),
        alpha_x=float_arg_text(alpha_x),
    )


def list_checkpoint_candidates(checkpoint_dir: Path, checkpoint_glob: str) -> List[Path]:
    if not checkpoint_dir.exists():
        return []
    return sorted(path.resolve() for path in checkpoint_dir.glob(checkpoint_glob) if path.is_file())


def choose_checkpoint(checkpoint_paths: Sequence[Path]) -> Tuple[Optional[Path], str]:
    if not checkpoint_paths:
        return None, ""
    if len(checkpoint_paths) == 1:
        return checkpoint_paths[0], ""
    chosen = max(checkpoint_paths, key=lambda path: path.stat().st_mtime)
    note = "multiple checkpoints found; using most recent: " + ",".join(str(path) for path in checkpoint_paths)
    return chosen, note


def command_text(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_tag(dataset: str, alpha_a_text: str, alpha_x_text: str, seed: int) -> str:
    return "{dataset}_aA_{alpha_a}_aX_{alpha_x}_seed_{seed}".format(
        dataset=dataset,
        alpha_a=alpha_a_text,
        alpha_x=alpha_x_text,
        seed=seed,
    )


def write_summary(rows: List[Dict[str, str]], summary_csv: Path, summary_json: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_index",
        "dataset",
        "alphaX",
        "alphaA",
        "gpu",
        "num_samples",
        "seed",
        "status",
        "returncode",
        "started_at",
        "finished_at",
        "checkpoint_dir",
        "checkpoint_path",
        "checkpoint_note",
        "save_pkl_dir",
        "save_pt_path",
        "log_path",
        "cmd",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def run_and_tee(cmd: Sequence[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_str = command_text(cmd)
    print("$ " + cmd_str)

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("CMD\n")
        handle.write(cmd_str + "\n\n")

        if dry_run:
            handle.write("[dry_run] command not executed.\n")
            print("[dry_run] command not executed.")
            return 0

        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)

        returncode = proc.wait()
        handle.write("\nRETURNCODE\n")
        handle.write(str(returncode) + "\n")
    return returncode


def main() -> int:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    sample_path = repo_dir / "sample.py"
    if not sample_path.exists():
        raise FileNotFoundError("sample.py not found: {path}".format(path=sample_path))

    batch_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.out_dir).resolve() / "{dataset}_{stamp}".format(
        dataset=args.dataset,
        stamp=batch_stamp,
    )
    logs_dir = run_dir / "logs"
    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"

    alpha_x_text = float_arg_text(args.alphaX)
    total_runs = len(args.alphaA_values)
    rows: List[Dict[str, str]] = []

    save_pkl_root = Path(args.save_pkl_root).resolve() if args.save_pkl_root is not None else None
    save_pt_root = Path(args.save_pt_root).resolve() if args.save_pt_root is not None else None
    if save_pt_root is not None:
        save_pt_root.mkdir(parents=True, exist_ok=True)

    print("repo_dir   : {repo_dir}".format(repo_dir=repo_dir))
    print("python_exec: {python_exec}".format(python_exec=args.python_exec))
    print("dataset    : {dataset}".format(dataset=args.dataset))
    print("alphaX     : {alpha_x}".format(alpha_x=alpha_x_text))
    print("alphaA list: {values}".format(values=", ".join(float_arg_text(v) for v in args.alphaA_values)))
    print("gpu        : {gpu}".format(gpu=args.gpu))
    print("num_samples: {num_samples}".format(num_samples=args.num_samples))
    print("seed       : {seed}".format(seed=args.seed))
    print("out_dir    : {out_dir}".format(out_dir=run_dir))
    if save_pkl_root is not None:
        print("save_pkl_root: {path}".format(path=save_pkl_root))
    if save_pt_root is not None:
        print("save_pt_root : {path}".format(path=save_pt_root))
    if args.extra_sample_args:
        print("extra args : {extra}".format(extra=command_text(args.extra_sample_args)))
    print("")

    exit_code = 0

    for index, alpha_a in enumerate(args.alphaA_values, start=1):
        alpha_a_text = float_arg_text(alpha_a)
        checkpoint_dir = repo_dir / checkpoint_dir_name(args.dataset, alpha_a, args.alphaX)
        checkpoint_candidates = list_checkpoint_candidates(checkpoint_dir, args.checkpoint_glob)
        checkpoint_path, checkpoint_note = choose_checkpoint(checkpoint_candidates)
        run_name = run_tag(args.dataset, alpha_a_text, alpha_x_text, args.seed)
        log_path = logs_dir / "{idx:02d}_{run_name}.log".format(idx=index, run_name=run_name)

        save_pkl_dir = save_pkl_root / run_name if save_pkl_root is not None else None
        save_pt_path = save_pt_root / "{run_name}.pyg.pt".format(run_name=run_name) if save_pt_root is not None else None

        row: Dict[str, str] = {
            "run_index": str(index),
            "dataset": args.dataset,
            "alphaX": alpha_x_text,
            "alphaA": alpha_a_text,
            "gpu": str(args.gpu),
            "num_samples": str(args.num_samples),
            "seed": str(args.seed),
            "status": "",
            "returncode": "",
            "started_at": "",
            "finished_at": "",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
            "checkpoint_note": checkpoint_note,
            "save_pkl_dir": str(save_pkl_dir) if save_pkl_dir is not None else "",
            "save_pt_path": str(save_pt_path) if save_pt_path is not None else "",
            "log_path": str(log_path),
            "cmd": "",
        }

        print("[{current}/{total}] alphaA={alpha_a}".format(
            current=index,
            total=total_runs,
            alpha_a=alpha_a_text,
        ))

        if checkpoint_path is None:
            row["status"] = "missing_checkpoint"
            row["returncode"] = ""
            row["started_at"] = datetime.now().isoformat(timespec="seconds")
            row["finished_at"] = row["started_at"]
            print("Checkpoint not found in {path}".format(path=checkpoint_dir))
            rows.append(row)
            write_summary(rows, summary_csv, summary_json)
            print("")
            if args.skip_missing:
                continue
            return 1

        cmd = [
            args.python_exec,
            "sample.py",
            "--model_path",
            str(checkpoint_path),
            "--num_samples",
            str(args.num_samples),
            "--gpu",
            str(args.gpu),
            "--seed",
            str(args.seed),
        ]
        if save_pkl_dir is not None:
            cmd.extend(["--save_pkl_dir", str(save_pkl_dir)])
        if save_pt_path is not None:
            cmd.extend(["--save_pt_path", str(save_pt_path)])
        if args.skip_internal_eval:
            cmd.append("--skip_internal_eval")
        cmd.extend(args.extra_sample_args)

        row["cmd"] = command_text(cmd)
        row["started_at"] = datetime.now().isoformat(timespec="seconds")
        returncode = run_and_tee(cmd, repo_dir, log_path, args.dry_run)
        row["finished_at"] = datetime.now().isoformat(timespec="seconds")
        row["returncode"] = str(returncode)
        row["status"] = "ok" if returncode == 0 else "failed"

        rows.append(row)
        write_summary(rows, summary_csv, summary_json)
        print("")

        if returncode != 0:
            exit_code = returncode or 1
            if args.stop_on_error:
                break

    print("summary csv : {path}".format(path=summary_csv))
    print("summary json: {path}".format(path=summary_json))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
