"""CPU tests of the opt-in pilot; CUDA and graph generation are never executed."""

import argparse
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import torch

import proxy_minimal_pilot as pilot
import train_controller


class ToyController(torch.nn.Module):
    def __init__(self, *, zero=False, mutate=False):
        super().__init__()
        self._denoise_fn = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        self.register_buffer("schedule", torch.tensor([0.2, 0.8]))
        self.fair_score_k_raw = torch.nn.Parameter(torch.zeros(2))
        self.fair_score_eta_raw = torch.nn.Parameter(torch.ones(2))
        self.zero, self.mutate, self.calls = zero, mutate, []
        self.requires_grad_(False)
        self.fair_score_k_raw.requires_grad_(True)
        self.fair_score_eta_raw.requires_grad_(True)
        self.eval()

    def sample(self, count, *, return_controller_replay):
        assert count == 2 and return_controller_replay is True
        assert not torch.is_grad_enabled() and not self.training
        self.calls.append(count)
        return object(), {"fixture": True}

    def compute_fair_controller_loss_from_replay(self, replay):
        assert replay == {"fixture": True}
        if self.mutate:
            self.schedule.add_(1)
        loss = ((self.fair_score_k_raw - 0.3).square()
                + (self.fair_score_eta_raw - 0.2).square()).sum()
        return loss * (0 if self.zero else 1), {"toy_loss": 1.0}

    def get_fair_controller_state_dict(self):
        return {"fair_score_k_raw": self.fair_score_k_raw.detach().clone(),
                "fair_score_eta_raw": self.fair_score_eta_raw.detach().clone(),
                "num_timesteps": 2}


def _args():
    return argparse.Namespace(**pilot.PILOT_CONTROLLER_CONFIG,
                              clip_value=1.0, clip_norm=None)


def _audit():
    return {"counts": {"optimizer_steps": 0, "replay_sample_calls": 0,
                       "replay_graphs": 0, "auto_export_graphs": 0,
                       "evaluation_graphs": 0, "gcn_fits": 0, "retries": 0}}


class PilotTrainingTest(unittest.TestCase):
    def test_fixed_configuration_and_physical_addressing(self):
        self.assertEqual(pilot.PILOT_CONTROLLER_CONFIG, {
            "controller_epochs": 100, "controller_lr": 5e-4,
            "fair_score_k": 0.5, "fair_score_eta": 0.005,
            "fair_score_fair_loss_weight": 1.0, "fair_score_utility_loss_weight": 0.1,
            "fair_score_k_tracking_loss_weight": 0.01,
            "controller_replay_num_samples": 2, "controller_replay_refresh": 10,
            "fair_score_guidance_normalize": True,
        })
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("torch.cuda.is_available", side_effect=AssertionError("CUDA forbidden")):
                pilot.validate_physical_gpu4("cuda:4")
            with self.assertRaisesRegex(ValueError, "physical cuda:4"):
                pilot.validate_physical_gpu4("cuda:0")
        for visibility in ("7", "0,1,2,3,4,5,6,7", ""):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visibility}):
                with self.assertRaisesRegex(ValueError, "Unset CUDA_VISIBLE_DEVICES"):
                    pilot.validate_physical_gpu4("cuda:4")

    def test_actual_saved_args_preserved_and_fixed_cli_rejects_overrides(self):
        cli = argparse.Namespace(
            device="cuda:4", seed=910001, fair_score_metric="sp",
            controller_pretrained_ckpt="../EDGE_fairness/wandb/cora/checkpoint_4999.pt",
            pilot_backbone_args="../EDGE_fairness/wandb/cora/stage1_fixed_eval_4999/args.pickle",
            pilot_graph="../EDGE_fairness/graphs/cora_feat.pkl", pilot_audit_path=None,
            controller_root="results/proxy_minimal/controllers", name="cora_prespecified_pilot_sp",
            diffusion_steps=1000, diffusion_dim=64,
        )
        # Keep the saved-argument contract independent of historical run assets.
        saved_fixture = argparse.Namespace(
            dataset="cora", diffusion_steps=256, diffusion_dim=128,
            num_heads=[8, 8, 8, 8, 1], degree=True, norm="None",
            noise_schedule="linear", loss_type="vb_ce_xt_prescribred_st",
            num_node_feat=1433, use_node_feat=True, max_degree=168,
            num_node_classes=0, augmented_features=[],
        )
        with tempfile.TemporaryDirectory(prefix="pilot_saved_args_") as temporary, \
                mock.patch.object(pilot, "ROOT", Path(temporary) / "EDGE_fairness_loss"), \
                mock.patch.dict(os.environ, {}, clear=True):
            pilot.ROOT.mkdir()
            for name in ("pilot_backbone_args", "pilot_graph", "controller_pretrained_ckpt"):
                path = pilot._path(getattr(cli, name))
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "pilot_backbone_args":
                    with path.open("wb") as stream:
                        pickle.dump(saved_fixture, stream)
                else:
                    path.touch()
            with mock.patch("torch.cuda.is_available", side_effect=AssertionError("CUDA forbidden")):
                result = pilot.prepare_pilot_args(cli)
            with pilot._path(cli.pilot_backbone_args).open("rb") as stream:
                saved = pickle.load(stream)
            for name in ("diffusion_steps", "diffusion_dim", "num_heads", "degree", "norm",
                         "noise_schedule", "loss_type", "num_node_feat", "use_node_feat",
                         "max_degree", "num_node_classes", "augmented_features"):
                self.assertEqual(getattr(result, name), getattr(saved, name))
            self.assertEqual(result.diffusion_steps, 256)
            self.assertEqual(result.diffusion_dim, 128)
            self.assertEqual(result.num_node_feat, 1433)
            self.assertEqual(result.fair_score_eta, 0.005)
            self.assertTrue(result.fair_score_guidance_normalize)
            self.assertEqual(result.device, "cuda:4")
            self.assertEqual(result.seed, 910001)
            with self.assertRaisesRegex(ValueError, "architecture/data"):
                pilot.prepare_pilot_args(cli, {"diffusion_steps"})
            cli.controller_epochs = 101
            with self.assertRaisesRegex(ValueError, "fixes controller_epochs"):
                pilot.prepare_pilot_args(cli, {"controller_epochs"})

    def test_frozen_cpu_training_exact_replay_budget_and_both_parameters_change(self):
        model, args, audit = ToyController(), _args(), _audit()
        before = pilot.snapshot_backbone(model)
        optimizer = torch.optim.Adam([model.fair_score_k_raw, model.fair_score_eta_raw], lr=args.controller_lr)
        rows = []
        loss, stats = pilot.train_pilot_epochs(args, model, optimizer, audit, log_row=rows.append)
        self.assertEqual(audit["counts"], {
            "optimizer_steps": 100, "replay_sample_calls": 10, "replay_graphs": 20,
            "auto_export_graphs": 0, "evaluation_graphs": 0, "gcn_fits": 0, "retries": 0})
        self.assertEqual(model.calls, [2] * 10)
        self.assertEqual(len(rows), 100)
        self.assertTrue(audit["backbone_unchanged"])
        self.assertEqual(audit["backbone_before"], audit["backbone_after"])
        self.assertEqual(audit["trainable_names"], sorted(pilot.CONTROLLER_KEYS))
        for name in pilot.CONTROLLER_KEYS:
            item = audit["controller_diagnostics"][name]
            self.assertEqual(item["grad_nonzero_epochs"], 100)
            self.assertEqual(item["grad_missing_epochs"], 0)
            self.assertTrue(item["grad_finite"] and item["param_changed"])
            self.assertGreater(item["param_delta_max"], 0)
        for name, value in pilot.snapshot_backbone(model).items():
            self.assertTrue(torch.equal(before[name], value))

    def test_zero_gradients_report_without_retry_or_search(self):
        model, args, audit = ToyController(zero=True), _args(), _audit()
        optimizer = torch.optim.Adam([model.fair_score_k_raw, model.fair_score_eta_raw], lr=args.controller_lr)
        pilot.train_pilot_epochs(args, model, optimizer, audit)
        self.assertEqual(audit["counts"]["optimizer_steps"], 100)
        self.assertEqual(audit["counts"]["replay_graphs"], 20)
        for item in audit["controller_diagnostics"].values():
            self.assertEqual(item["grad_nonzero_epochs"], 0)
            self.assertFalse(item["param_changed"])

    def test_nonfinite_gradient_stops_and_audits_without_retry(self):
        model, args, audit = ToyController(), _args(), _audit()
        model.fair_score_eta_raw.register_hook(lambda gradient: gradient * float("nan"))
        optimizer = torch.optim.Adam([model.fair_score_k_raw, model.fair_score_eta_raw], lr=args.controller_lr)
        with self.assertRaisesRegex(RuntimeError, "Nonfinite fair_score_eta_raw gradient"):
            pilot.train_pilot_epochs(args, model, optimizer, audit)
        self.assertEqual(audit["counts"]["optimizer_steps"], 0)
        self.assertEqual(audit["counts"]["replay_graphs"], 2)
        self.assertFalse(audit["controller_diagnostics"]["fair_score_eta_raw"]["grad_finite"])
        self.assertTrue(audit["backbone_unchanged"])

    def test_backbone_buffer_mutation_is_detected(self):
        model, args, audit = ToyController(mutate=True), _args(), _audit()
        optimizer = torch.optim.Adam([model.fair_score_k_raw, model.fair_score_eta_raw], lr=args.controller_lr)
        with self.assertRaisesRegex(RuntimeError, "Backbone tensor changed"):
            pilot.train_pilot_epochs(args, model, optimizer, audit)
        self.assertFalse(audit["backbone_unchanged"])
        self.assertNotEqual(audit["backbone_before"]["sha256"], audit["backbone_after"]["sha256"])

    def test_cli_pilot_branches_before_legacy_data_or_evaluation(self):
        original_import = builtins.__import__
        def no_legacy_import(name, *args, **kwargs):
            if name in {"experiment", "datasets.data", "diffusion.experiment"} or name.startswith("tensorflow"):
                raise AssertionError(f"Pilot imported {name}")
            return original_import(name, *args, **kwargs)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("builtins.__import__", side_effect=no_legacy_import))
            prepare = stack.enter_context(mock.patch.object(pilot, "prepare_pilot_args", return_value="prepared"))
            run = stack.enter_context(mock.patch.object(pilot, "run_pilot_training", return_value="audit"))
            for name in ("get_data", "set_seeds", "evaluate_controller", "save_generated_graphs_for_lp"):
                stack.enter_context(mock.patch.object(train_controller, name, side_effect=AssertionError(name)))
            result = train_controller.main(["--prespecified_pilot", "--device", "cuda:4"])
            self.assertEqual(result, "audit")
            run.assert_called_once_with("prepared")
            self.assertEqual(prepare.call_args.kwargs["supplied_options"], {"prespecified_pilot", "device"})

    def test_default_cli_still_uses_legacy_path(self):
        with mock.patch.object(pilot, "run_pilot_training", side_effect=AssertionError("pilot forbidden")):
            with mock.patch.object(train_controller, "prepare_args", side_effect=RuntimeError("legacy path reached")):
                with self.assertRaisesRegex(RuntimeError, "legacy path reached"):
                    train_controller.main([])

    def test_full_orchestration_saves_only_final_and_refuses_second_run(self):
        temp_root = pilot.ROOT / "results/proxy_minimal"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pilot_training_cpu_", dir=temp_root) as tmp:
            args, model = _args(), ToyController()
            args.controller_root = tmp
            args.fair_score_metric = "sp"
            args.name = "toy"
            args.device, args.seed = "cuda:4", 910001
            args.pilot_audit_path = str(Path(tmp) / "sp/toy/pilot_training_audit.json")
            args.controller_pretrained_ckpt = "synthetic_backbone"
            args.pilot_backbone_args = "synthetic_args"
            args.pilot_graph = "synthetic_graph"
            real_record = pilot.file_record
            def record(path):
                if str(path).startswith("synthetic_"):
                    return {"path": path, "sha256": "synthetic", "bytes": 0}
                return real_record(path)
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                stack.enter_context(mock.patch.object(pilot, "file_record", side_effect=record))
                load = stack.enter_context(mock.patch.object(pilot, "_load_pilot_model", return_value=(
                    model, [model.fair_score_k_raw, model.fair_score_eta_raw])))
                stack.enter_context(mock.patch("torch.cuda.is_available", return_value=True))
                stack.enter_context(mock.patch("torch.cuda.device_count", return_value=5))
                # Adam may inspect capture state after earlier tests imported
                # CUDA-aware modules; never let that query reach a real device.
                stack.enter_context(mock.patch("torch.cuda.is_current_stream_capturing", return_value=False))
                select = stack.enter_context(mock.patch("torch.cuda.set_device"))
                stack.enter_context(mock.patch("torch.cuda.reset_peak_memory_stats"))
                stack.enter_context(mock.patch("torch.cuda.max_memory_allocated", return_value=0))
                stack.enter_context(mock.patch("torch.cuda.max_memory_reserved", return_value=0))
                stack.enter_context(mock.patch("torch.cuda.get_device_name", return_value="CPU toy fixture"))
                stack.enter_context(mock.patch("torch.cuda.device", return_value=contextlib.nullcontext()))
                seed = stack.enter_context(mock.patch("torch.cuda.manual_seed"))
                sync = stack.enter_context(mock.patch("torch.cuda.synchronize"))
                for name in ("evaluate_controller", "save_generated_graphs_for_lp", "get_data"):
                    stack.enter_context(mock.patch.object(train_controller, name, side_effect=AssertionError(name)))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                audit = pilot.run_pilot_training(args)
                with self.assertRaisesRegex(FileExistsError, "overwrite/retrain"):
                    pilot.run_pilot_training(args)
            self.assertEqual(load.call_count, 1)
            select.assert_called_once_with(4)
            seed.assert_called_once_with(910001)
            sync.assert_called_once_with(4)
            self.assertEqual(audit["status"], "complete")
            self.assertEqual(audit["final_epoch"], 99)
            self.assertFalse(audit["test_used"])
            self.assertEqual(audit["counts"]["auto_export_graphs"], 0)
            run_dir = Path(tmp) / "sp/toy"
            self.assertEqual([path.name for path in (run_dir / "check").iterdir()], ["controller_final.pt"])
            checkpoint = torch.load(run_dir / "check/controller_final.pt", weights_only=False, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], 99)
            self.assertNotIn("model", checkpoint)
            with (run_dir / "args.pickle").open("rb") as stream:
                self.assertEqual(checkpoint["args"], vars(pickle.load(stream)))
            persisted = json.loads((run_dir / "pilot_training_audit.json").read_text())
            self.assertEqual(persisted["final_checkpoint"], real_record(run_dir / "check/controller_final.pt"))


if __name__ == "__main__":
    unittest.main()
