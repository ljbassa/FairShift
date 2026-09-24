"""Exercise metric selection and real artifact paths without importing DGL."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fairness_options import metric_directory, metric_file, resolve_controller_metric


def load_functions(filename, names):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"Path": Path, "os": os, "json": json, "metric_directory": metric_directory}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def run_script(script, *args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )


def test_checkpoint_metric_defaults_and_conflict():
    assert resolve_controller_metric(None) == "sp"
    assert resolve_controller_metric(None, "eo") == "eo"
    assert resolve_controller_metric("eo") == "eo"
    with pytest.raises(ValueError, match="checkpoint uses"):
        resolve_controller_metric("sp", "eo")


def test_output_paths_do_not_collide():
    assert metric_directory("runs", "sp") != metric_directory("runs", "eo")
    assert metric_directory("runs/eo", "eo") == Path("runs/eo")
    assert metric_file("runs/result.pt", "eo") == Path("runs/eo/result.pt")


def test_unguided_sample_tag_is_preserved():
    functions = load_functions("sample.py", {"sanitize_tag_value", "build_sample_tag"})
    args = SimpleNamespace(model_path="base.pt", seed=0, sp_eta=.01, sp_k=.15)
    assert functions["build_sample_tag"](args, False) == "base_seed0_eta0p01_k0p15_ctrl0"
    args.fair_score_sp = True
    assert functions["build_sample_tag"](args, False).endswith("_sp")
    args.fair_score_metric = "eo"
    assert functions["build_sample_tag"](args, True).endswith("_eo")


@pytest.mark.parametrize("explicit", [False, True])
def test_training_paths_and_latest_manifests(tmp_path, explicit):
    functions = load_functions("train_controller.py", {"make_log_dir", "write_latest_run_manifest"})
    dirs = []
    for metric in ("sp", "eo"):
        args = SimpleNamespace(
            out_dir=str(tmp_path / "custom") if explicit else None,
            log_home=None if explicit else str(tmp_path), name="same_run",
            fair_score_metric=metric, controller_pretrained_ckpt="stage1.pt",
            num_generation=2, device="cpu",
        )
        log_dir, check_dir = functions["make_log_dir"](args, "cora")
        log_dir = Path(log_dir)
        dirs.append(log_dir)
        manifest_dir = log_dir if explicit else log_dir.parent
        manifest = json.loads((manifest_dir / "latest_run.json").read_text())
        assert manifest["fair_score_metric"] == metric
        assert f"--fair_score_metric {metric}" in manifest["sample_command"]
        assert Path(check_dir).is_dir()
    assert dirs[0] != dirs[1]
    assert (dirs[0] / "args.json").exists()


def test_grid_forwards_metric_and_scopes_manifest(tmp_path):
    controller_root = tmp_path / "controller"
    run_script(
        "scripts/run_controller_grid.py", "--stage1_ckpt", "unused.pt",
        "--controller_root", controller_root, "--fair_score_metric", "eo",
        "--manifest", tmp_path / "grid.jsonl", "--dry_run", "--max_runs", "1",
    )
    record = json.loads((tmp_path / "eo/grid.jsonl").read_text().strip())
    assert record["fair_score_metric"] == "eo"
    cmd = record["command"]
    assert cmd[cmd.index("--fair_score_metric") + 1] == "eo"
    assert cmd[cmd.index("--controller_root") + 1] == str(controller_root / "eo")
    assert not (tmp_path / "sp/grid.jsonl").exists()


@pytest.mark.parametrize("checkpoint,dataset_override,expected_dataset", [
    ("citeseer_0.0_0.0_cpts/Sync_T3.pth", None, "citeseer"),
    ("amazon_photo_0.0_0.0_cpts/Sync_T3.pth", None, "amazon_photo"),
    ("custom/checkpoint.pth", "pokec_n", "pokec_n"),
])
def test_grid_resolves_non_cora_output_root(tmp_path, checkpoint, dataset_override, expected_dataset):
    extra = ["--dataset", dataset_override] if dataset_override else []
    run_script(
        "scripts/run_controller_grid.py", "--stage1_ckpt", checkpoint,
        "--log_home", tmp_path, "--name_prefix", "dataset_check",
        "--fair_score_metric", "eo", "--dry_run", "--max_runs", "1", *extra,
    )
    root = tmp_path / expected_dataset / "Sync/controller/eo"
    record = json.loads((root / "dataset_check_manifest.jsonl").read_text().strip())
    command = record["command"]
    assert command[command.index("--controller_root") + 1] == str(root)
    assert not (tmp_path / "cora").exists()


@pytest.mark.parametrize("title", [None, "Feature EO comparison"])
def test_grid_pareto_title_preserves_feature_export_options(tmp_path, monkeypatch, title):
    import runpy

    module = runpy.run_path(str(ROOT / "scripts/run_controller_grid.py"))
    argv = [
        "run_controller_grid.py", "--stage1_ckpt", "unused.pth",
        "--fair_score_metric", "eo", "--pareto_front_csv", "custom/front.csv",
        "--pareto_extra_summary_csv", "previous.csv",
    ]
    if title is not None:
        argv += ["--pareto_title", title]
    monkeypatch.setattr(sys, "argv", argv)
    args = module["parse_args"]()
    commands = []

    def capture_command(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", capture_command)
    module["run_generated_pareto"](args, tmp_path, tmp_path / "controller/eo", "citeseer")
    plot_command = commands[1]
    assert plot_command[plot_command.index("--title") + 1] == (title or "citeseer: Controller LP Pareto")
    assert plot_command[plot_command.index("--front_csv") + 1] == str(tmp_path / "custom/eo/front.csv")
    assert plot_command[plot_command.index("--extra_summary_csv") + 1] == str(tmp_path / "previous.csv")
    assert plot_command[plot_command.index("--x_metric") + 1] == "lp/eo_abs_gap_mean"


def test_summary_and_plot_keep_metrics_separate(tmp_path):
    import csv

    root = tmp_path / "controller"
    for metric, eo_gap in (("sp", 0.9), ("eo", 0.05)):
        run = root / metric / "shared_name"
        (run / "check").mkdir(parents=True)
        (run / "generated_samples").mkdir()
        torch.save({"args": {"fair_score_metric": metric}}, run / "check/controller_last.pt")
        (run / "controller_metrics.jsonl").write_text('{"epoch": 1, "loss": 0.1}\n')
        (run / "generated_samples/generated.summary.csv").write_text(
            f"lp/auc_mean,lp/eo_abs_gap_mean,lp/score_sp_abs_gap_mean\n0.8,{eo_gap},0.2\n"
        )
    requested_summary = tmp_path / "summary.csv"
    run_script(
        "scripts/summarize_controller_grid.py", "--controller_root", root,
        "--prefix", "shared", "--fair_score_metric", "eo", "--out_csv", requested_summary,
    )
    summary = tmp_path / "eo/summary.csv"
    with summary.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["fair_score_metric"] == "eo"
    assert float(rows[0]["lp/eo_abs_gap_mean"]) == 0.05
    run_script(
        "scripts/plot_controller_grid_pareto.py", "--summary_csv", summary,
        "--fair_score_metric", "eo", "--out_path", tmp_path / "pareto.png",
        "--front_csv", tmp_path / "front.csv", "--label_points", "none",
    )
    assert (tmp_path / "eo/pareto.png").exists()
    assert (tmp_path / "eo/front.csv").exists()
    assert not (tmp_path / "pareto.png").exists()
