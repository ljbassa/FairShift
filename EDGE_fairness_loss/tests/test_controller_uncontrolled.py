"""CPU tests for the grid's unguided Stage-1 export and shared export seed."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import train_controller


class UncontrolledExportTest(unittest.TestCase):
    def test_uncontrolled_exports_stage1_without_constructing_optimizer(self):
        model = mock.Mock(fair_score_controller_train=False)
        model.to.return_value = model
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(train_controller, "set_seeds"))
            stack.enter_context(mock.patch.object(train_controller, "get_data"))
            stack.enter_context(mock.patch.object(
                train_controller, "prepare_data_args", return_value=(None,) * 7,
            ))
            stack.enter_context(mock.patch.object(train_controller, "get_model", return_value=model))
            load = stack.enter_context(mock.patch.object(train_controller, "load_pretrained_model"))
            log = stack.enter_context(mock.patch.object(
                train_controller, "make_log_dir", return_value=(tmp, str(Path(tmp) / "check")),
            ))
            export = stack.enter_context(mock.patch.object(train_controller, "save_generated_graphs_for_lp"))
            optimizer = stack.enter_context(mock.patch.object(
                train_controller.torch.optim, "Adam", side_effect=AssertionError("baseline must not train"),
            ))
            train_controller.main([
                "--uncontrolled", "--dataset", "cora", "--device", "cpu",
                "--controller_pretrained_ckpt", "stage1.pt", "--fair_score_metric", "eo",
                "--fair_score_controller_train", "--generation_seed", "42", "--num_generation", "8",
            ])
            args = log.call_args.args[0]
            for name in ("fair_score_sp", "fair_score_learn_k", "fair_score_learn_eta", "fair_score_controller_train"):
                self.assertFalse(getattr(args, name), name)
            self.assertEqual(args.generation_seed, 42)
            load.assert_called_once_with(model, "stage1.pt", "cpu")
            optimizer.assert_not_called()
            model.freeze_for_fair_controller_training.assert_not_called()
            export.assert_called_once_with(
                args=args, model=model, log_dir=tmp, tag="controller_best", epoch=None,
                checkpoint_path="stage1.pt",
            )

    def test_uncontrolled_refuses_model_with_guidance_enabled(self):
        model = mock.Mock(fair_score_controller_train=True)
        model.to.return_value = model
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(train_controller, "set_seeds"))
            stack.enter_context(mock.patch.object(train_controller, "get_data"))
            stack.enter_context(mock.patch.object(
                train_controller, "prepare_data_args", return_value=(None,) * 7,
            ))
            stack.enter_context(mock.patch.object(train_controller, "get_model", return_value=model))
            stack.enter_context(mock.patch.object(train_controller, "load_pretrained_model"))
            export = stack.enter_context(mock.patch.object(train_controller, "save_generated_graphs_for_lp"))
            with self.assertRaisesRegex(RuntimeError, "guidance to be disabled"):
                train_controller.main([
                    "--uncontrolled", "--dataset", "cora", "--device", "cpu",
                    "--controller_pretrained_ckpt", "stage1.pt",
                ])
            export.assert_not_called()
            model.freeze_for_fair_controller_training.assert_not_called()

    def test_both_arms_reset_export_seed_independently_of_training_rng(self):
        observed = []

        def sample(args, model, total):
            self.assertEqual(total, 8)
            observed.append(torch.rand(3))
            return [torch.tensor(i) for i in range(total)]

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            train_controller, "sample_controller_pyg_graphs", side_effect=sample,
        ), mock.patch.object(torch.cuda, "is_available", return_value=False):
            for uncontrolled in (True, False):
                torch.rand(13 if uncontrolled else 127)
                args = argparse.Namespace(
                    num_generation=8, generation_seed=42, uncontrolled=uncontrolled,
                    dataset="cora", device="cpu", fair_score_metric="eo", fair_score_eo_min_mass=1e-6,
                )
                model = mock.Mock(fair_score_controller_train=not uncontrolled)
                tag = "baseline" if uncontrolled else "controller"
                train_controller.save_generated_graphs_for_lp(
                    args, model, tmp, tag, None if uncontrolled else 2, "stage1.pt",
                )
                meta = json.loads((Path(tmp) / "generated_samples" / f"{tag}.meta.json").read_text())
                self.assertEqual(meta["generation_seed"], 42)
                self.assertEqual(meta["num_graphs"], 8)
                self.assertEqual(meta["uncontrolled"], uncontrolled)
                self.assertEqual(meta["fairness_sampling_enabled"], not uncontrolled)
            expected = torch.rand(3, generator=torch.Generator().manual_seed(42))
            torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
            torch.testing.assert_close(observed[1], expected, rtol=0, atol=0)

    def test_legacy_export_does_not_reset_random_seed(self):
        args = argparse.Namespace(
            num_generation=8, dataset="cora", device="cpu",
            fair_score_metric="eo", fair_score_eo_min_mass=1e-6,
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            train_controller, "set_seeds",
        ) as seed, mock.patch.object(train_controller, "sample_controller_pyg_graphs", return_value=[None] * 8):
            train_controller.save_generated_graphs_for_lp(
                args, mock.Mock(fair_score_controller_train=True), tmp, "controller_best", 2, "controller.pt",
            )
            seed.assert_not_called()
            meta = json.loads((Path(tmp) / "generated_samples" / "controller_best.meta.json").read_text())
            self.assertIsNone(meta["generation_seed"])
            self.assertFalse(meta["uncontrolled"])
            self.assertTrue(meta["fairness_sampling_enabled"])

    def test_pilot_rejects_grid_only_modes(self):
        with mock.patch("proxy_minimal_pilot.run_pilot_training") as pilot:
            for extra in (["--uncontrolled"], ["--generation_seed", "42"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        train_controller.main(["--prespecified_pilot"] + extra)
                    self.assertEqual(error.exception.code, 2)
            pilot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
