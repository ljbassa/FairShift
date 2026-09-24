"""CPU-only coverage of baseline exports and resuming generated-graph evaluation."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import run_controller_grid as grid


class ControllerGridBaselineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.controller_root = self.repo / "controller" / "eo"
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def args(self, *extra):
        argv = [
            "run_controller_grid.py", "--repo_dir", str(self.repo),
            "--stage1_ckpt", "check/stage1.pt", "--controller_root", "controller",
            "--name_prefix", "test_eo_grid", "--fair_score_metric", "eo",
            "--device", "cpu", "--generation_seed", "42", "--num_generation", "8",
            "--max_runs", "1",
        ]
        with mock.patch("sys.argv", argv + list(extra)):
            return grid.parse_args()

    def run_name(self, args):
        return grid.make_run_name(
            args.name_prefix, args.eta_values[0], args.controller_lrs[0],
            args.fair_weights[0], args.utility_weights[0], args.k_tracking_weights[0],
            args.fair_score_k_values[0], args.fair_score_guidance_normalize,
        )

    def existing_controller(self, args):
        run_dir = self.controller_root / self.run_name(args)
        checkpoint = run_dir / "check" / "controller_final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        graph = run_dir / "generated_samples" / "controller_best.pyg_full.pt"
        graph.parent.mkdir(parents=True)
        graph.touch()
        return graph

    def command_value(self, command, flag):
        return command[command.index(flag) + 1]

    def test_include_uncontrolled_exports_separate_baseline_from_same_checkpoint(self):
        args = self.args("--include_uncontrolled", "--max_runs", "0")
        with mock.patch.object(grid, "parse_args", return_value=args), mock.patch.object(
            grid.subprocess, "run", return_value=mock.Mock(returncode=0),
        ) as run:
            grid.main()
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[1], "train_controller.py")
        self.assertIn("--uncontrolled", command)
        baseline_name = self.command_value(command, "--name")
        self.assertEqual(baseline_name, "uncontrolled_test_eo_grid")
        self.assertFalse(baseline_name.startswith(args.name_prefix))
        self.assertEqual(self.command_value(command, "--controller_pretrained_ckpt"), args.stage1_ckpt)
        self.assertEqual(self.command_value(command, "--generation_seed"), "42")
        self.assertEqual(self.command_value(command, "--num_generation"), "8")
        self.assertEqual(self.command_value(command, "--fair_score_metric"), "eo")
        self.assertEqual(run.call_args.kwargs["cwd"], self.repo)

    def test_dry_run_starts_neither_baseline_training_nor_evaluation(self):
        args = self.args("--include_uncontrolled", "--run_generated_eval", "--dry_run")
        with mock.patch.object(grid, "parse_args", return_value=args), mock.patch.object(
            grid.subprocess, "run",
        ) as run:
            grid.main()
        run.assert_not_called()
        manifest = self.controller_root / "test_eo_grid_manifest.jsonl"
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "dry_run")
        self.assertEqual(self.command_value(records[0]["command"], "--generation_seed"), "42")

    def test_generated_evaluation_forwards_seed_and_device(self):
        args = self.args("--generated_eval_seed", "73", "--generated_eval_device", "cpu")
        graph = self.existing_controller(args)
        with mock.patch.object(grid.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            summary, status = grid.run_generated_eval(args, self.repo, self.controller_root, self.run_name(args))
        self.assertEqual(status, 0)
        self.assertEqual(summary, graph.with_name("controller_best.pyg_full.overlap_lp_gae_summary.csv"))
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[1], "evaluate_generated_graphs.py")
        self.assertEqual(self.command_value(command, "--seed"), "73")
        self.assertEqual(self.command_value(command, "--device"), "cpu")
        self.assertEqual(self.command_value(command, "--graph_path"), str(graph))

    def test_missing_generated_graph_is_failure_without_starting_evaluator(self):
        args = self.args()
        with mock.patch.object(grid.subprocess, "run") as run:
            result = grid.run_generated_eval(args, self.repo, self.controller_root, self.run_name(args))
        self.assertEqual(result, (None, 1))
        run.assert_not_called()

    def test_skip_existing_retries_missing_evaluation_without_retraining(self):
        args = self.args("--skip_existing", "--run_generated_eval", "--skip_generated_pareto")
        graph = self.existing_controller(args)
        with mock.patch.object(grid, "parse_args", return_value=args), mock.patch.object(
            grid.subprocess, "run", return_value=mock.Mock(returncode=0),
        ) as run:
            grid.main()
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[1], "evaluate_generated_graphs.py")
        self.assertEqual(self.command_value(command, "--graph_path"), str(graph))
        record = json.loads((self.controller_root / "test_eo_grid_manifest.jsonl").read_text())
        self.assertEqual(record["status"], "skipped_existing")
        self.assertEqual(record["generated_eval_returncode"], 0)

    def test_skip_existing_dry_run_does_not_retry_evaluation(self):
        args = self.args("--skip_existing", "--run_generated_eval", "--dry_run", "--include_uncontrolled")
        self.existing_controller(args)
        with mock.patch.object(grid, "parse_args", return_value=args), mock.patch.object(
            grid.subprocess, "run",
        ) as run:
            grid.main()
        run.assert_not_called()

    def test_skip_existing_rejects_mismatched_uncontrolled_metadata(self):
        args = self.args("--include_uncontrolled", "--skip_existing")
        generated = self.controller_root / "uncontrolled_test_eo_grid" / "generated_samples"
        generated.mkdir(parents=True)
        (generated / "controller_best.pyg_full.pt").touch()
        valid_meta = {
            "checkpoint_path": "check/stage1.pt", "uncontrolled": True,
            "fairness_sampling_enabled": False, "generation_seed": 42, "num_graphs": 8,
        }
        for field, value in (
            ("checkpoint_path", "other/stage1.pt"), ("uncontrolled", False),
            ("fairness_sampling_enabled", True), ("generation_seed", 7), ("num_graphs", 3),
        ):
            with self.subTest(field=field):
                meta = dict(valid_meta, **{field: value})
                (generated / "controller_best.meta.json").write_text(json.dumps(meta))
                with mock.patch.object(grid.subprocess, "run") as run:
                    with self.assertRaisesRegex(ValueError, "metadata does not match"):
                        grid.run_uncontrolled(args, self.repo, self.controller_root)
                run.assert_not_called()

    def test_matching_uncontrolled_metadata_reuses_graph_and_retries_evaluation(self):
        args = self.args("--include_uncontrolled", "--skip_existing", "--run_generated_eval")
        generated = self.controller_root / "uncontrolled_test_eo_grid" / "generated_samples"
        generated.mkdir(parents=True)
        graph = generated / "controller_best.pyg_full.pt"
        graph.touch()
        (generated / "controller_best.meta.json").write_text(json.dumps({
            "checkpoint_path": str(self.repo / "check" / "stage1.pt"), "uncontrolled": True,
            "fairness_sampling_enabled": False, "generation_seed": 42, "num_graphs": 8,
        }))
        with mock.patch.object(grid.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            _, status = grid.run_uncontrolled(args, self.repo, self.controller_root)
        self.assertEqual(status, 0)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[1], "evaluate_generated_graphs.py")
        self.assertEqual(self.command_value(command, "--graph_path"), str(graph))


if __name__ == "__main__":
    unittest.main()
