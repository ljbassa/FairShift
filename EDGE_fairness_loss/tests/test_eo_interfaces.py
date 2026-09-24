import argparse
import csv
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from diffusion.diffusion_binomial_active import BinomialDiffusionActive
from evaluate_generated_graphs import samplepy_group_fairness_details
from model import add_model_args
from scripts import run_controller_grid
from train_controller import make_log_dir


class EOInterfaceTest(unittest.TestCase):
    def test_eo_scores_condition_on_only_positive_pairs(self):
        labels = np.asarray([1, 1, 1, 1, 0, 0])
        scores = np.asarray([0.9, 0.7, 0.4, 0.2, 0.8, 0.1])
        same = np.asarray([True, True, False, False, True, False])
        details = samplepy_group_fairness_details(labels, scores, same)
        self.assertAlmostEqual(details["eo_signed"], 0.5)
        self.assertAlmostEqual(details["eo_abs"], 0.5)
        self.assertEqual(details["eo_defined"], 1.0)
        scores[-2:] = [0.0, 1.0]
        changed = samplepy_group_fairness_details(labels, scores, same)
        self.assertEqual(changed["eo_signed"], details["eo_signed"])
        self.assertNotEqual(changed["sp_signed"], details["sp_signed"])

    def test_eo_missing_positive_group_is_undefined(self):
        details = samplepy_group_fairness_details(
            np.asarray([1, 0]), np.asarray([0.9, 0.2]), np.asarray([True, False])
        )
        self.assertEqual(details["eo_defined"], 0.0)
        self.assertTrue(np.isnan(details["eo_abs"]))

    def test_model_cli_defaults_to_sp_and_accepts_eo(self):
        parser = argparse.ArgumentParser()
        add_model_args(parser)
        self.assertEqual(parser.parse_args([]).fair_score_metric, "sp")
        self.assertEqual(parser.parse_args(["--fair_score_metric", "eo"]).fair_score_metric, "eo")

    def test_controller_checkpoint_restores_metric_and_old_defaults(self):
        model = BinomialDiffusionActive.__new__(BinomialDiffusionActive)
        torch.nn.Module.__init__(model)
        model.num_timesteps = 2
        model.fair_score_k_raw = torch.nn.Parameter(torch.zeros(2))
        model.fair_score_eta_raw = torch.nn.Parameter(torch.zeros(2))
        state = {
            "fair_score_metric": "eo",
            "fair_score_eo_min_mass": 0.025,
            "fair_score_k_raw": torch.tensor([0.1, 0.2]),
            "fair_score_eta_raw": torch.tensor([0.3, 0.4]),
        }
        model.load_fair_controller_state_dict({"controller": state})
        self.assertEqual(model.fair_score_metric, "eo")
        self.assertEqual(model.fair_score_eo_min_mass, 0.025)
        torch.testing.assert_close(model.fair_score_eta_raw, state["fair_score_eta_raw"])
        state.pop("fair_score_metric")
        state.pop("fair_score_eo_min_mass")
        model.load_fair_controller_state_dict(state)
        self.assertEqual(model.fair_score_metric, "sp")
        self.assertEqual(model.fair_score_eo_min_mass, 1e-6)

    def test_train_and_grid_keep_same_named_results_separate(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "controller"
            paths = []
            for metric in ("sp", "eo"):
                args = argparse.Namespace(log_home=None, controller_root=str(root), name="same_name", fair_score_metric=metric)
                with mock.patch("train_controller.get_args_table", return_value="args"):
                    log_dir, _ = make_log_dir(args, "cora", "multinomial_diffusion")
                paths.append(log_dir)
                with (Path(log_dir) / "args.pickle").open("rb") as stream:
                    self.assertEqual(pickle.load(stream).fair_score_metric, metric)
                subprocess.run([
                    sys.executable, "-B", str(repo / "scripts/run_controller_grid.py"),
                    "--repo_dir", str(repo), "--stage1_ckpt", "unused.pt",
                    "--controller_root", str(root), "--fair_score_metric", metric,
                    "--name_prefix", "same_grid", "--max_runs", "1", "--dry_run",
                ], check=True, stdout=subprocess.PIPE, text=True)
                manifest = root / metric / "same_grid_manifest.jsonl"
                record = json.loads(manifest.read_text())
                self.assertEqual(record["fair_score_metric"], metric)
                command = record["command"]
                self.assertEqual(command[command.index("--fair_score_metric") + 1], metric)
                self.assertEqual(command[command.index("--controller_root") + 1], str(root))
                args.controller_root = str(root / metric)
                with mock.patch("train_controller.get_args_table", return_value="args"):
                    self.assertEqual(make_log_dir(args, "cora", "multinomial_diffusion")[0], log_dir)
            self.assertNotEqual(paths[0], paths[1])

    def test_explicit_grid_outputs_stay_separate_and_plot_eo(self):
        repo = Path(__file__).resolve().parents[1]
        with mock.patch.object(sys, "argv", [
            "grid", "--stage1_ckpt", "unused.pt", "--fair_score_metric", "eo",
            "--pareto_summary_csv", "shared/summary.csv", "--pareto_plot_path", "shared/plot.jpg",
        ]):
            args = run_controller_grid.parse_args()
        with mock.patch.object(run_controller_grid.subprocess, "run", return_value=argparse.Namespace(returncode=0)) as run:
            summary, plot, result = run_controller_grid.run_generated_pareto(args, repo, repo / "controller/eo")
        self.assertEqual(summary, repo / "shared/eo/summary.csv")
        self.assertEqual(plot, repo / "shared/eo/plot.jpg")
        self.assertEqual(result, 0)
        command = run.call_args_list[-1].args[0]
        self.assertEqual(command[command.index("--x_metric") + 1], "lp/eo_abs_gap_mean")

    def test_summary_and_plot_read_only_selected_controller_metric(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "controller"
            for metric in ("sp", "eo"):
                run_dir = root / metric / "run_same"
                samples = run_dir / "generated_samples"
                samples.mkdir(parents=True)
                (run_dir / "controller_metrics.jsonl").write_text(json.dumps({"fair_score_metric": metric, "loss": 0.1}) + "\n")
                (samples / "lp_summary.csv").write_text("lp/auc_mean,lp/score_sp_abs_gap_mean,lp/eo_abs_gap_mean\n0.8,0.2,0.1\n")
                subprocess.run([
                    sys.executable, "-B", str(repo / "scripts/summarize_controller_grid.py"),
                    "--controller_root", str(root), "--fair_score_metric", metric, "--prefix", "run_",
                    "--out_csv", str(root / "shared_summary.csv"),
                ], check=True, stdout=subprocess.PIPE, text=True)
                summary = root / metric / "shared_summary.csv"
                with summary.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["fair_score_metric"], metric)
                other = dict(rows[0], fair_score_metric="eo" if metric == "sp" else "sp")
                with summary.open("a", newline="") as stream:
                    csv.DictWriter(stream, fieldnames=list(other)).writerow(other)
                result = subprocess.run([
                    sys.executable, "-B", str(repo / "scripts/plot_controller_grid_pareto.py"),
                    "--summary_csv", str(summary), "--fair_score_metric", metric,
                    "--out_path", str(root / "shared_plot.jpg"), "--front_csv", str(root / "shared_front.csv"),
                ], check=True, stdout=subprocess.PIPE, text=True)
                self.assertIn("valid points: 1", result.stdout)
                self.assertEqual(len(list((root / metric).glob("*.jpg"))), 1)
                with (root / metric / "shared_front.csv").open() as stream:
                    front = list(csv.DictReader(stream))
                self.assertEqual([row["fair_score_metric"] for row in front], [metric])


if __name__ == "__main__":
    unittest.main()
