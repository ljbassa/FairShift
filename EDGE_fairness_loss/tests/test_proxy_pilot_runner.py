"""CPU-only pilot orchestration checks; sampling, fits and CUDA are mocked."""

import contextlib
from copy import deepcopy
import csv
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch_geometric.data import Batch, Data

import proxy_minimal_metrics as metrics
from proxy_minimal_gcn import EmbeddingDecoder
from proxy_minimal_pilot import (PILOT_CONTROLLER_CONFIG, PILOT_GCN_CONFIG,
                                 PILOT_SMOKE_SEEDS, PILOT_TRAIN_SEEDS)
import run_proxy_minimal as runner


TEMP_ROOT = Path(__file__).resolve().parents[1] / "results/proxy_minimal"


def pilot_spec(output):
    selection = {"basis": "prespecified_pilot", "uses_test_for_selection": False}
    return {
        "policy": "prespecified_pilot", "dataset": "cora", "device": "cuda:4",
        "decode_chunk_size": 5, "root_seeds": list(range(930001, 930031)),
        "backbone_args": str(output / "reference_args.pickle"),
        "backbone_checkpoint": str(output / "reference_checkpoint.pt"),
        "graph": str(output / "reference_graph.pkl"),
        "controllers": {target: {
            "args": str(output / "controllers" / target / f"cora_prespecified_pilot_{target}" / "args.pickle"),
            "checkpoint": str(output / "controllers" / target / f"cora_prespecified_pilot_{target}" / "check/controller_final.pt"),
            "training_audit": str(output / "controllers" / target / f"cora_prespecified_pilot_{target}" / "pilot_training_audit.json"),
            "selection": {**selection, "checkpoint_rule": "final_epoch"},
        } for target in ("sp", "eo")},
        "gcn": {"config": dict(PILOT_GCN_CONFIG), "selection": dict(selection)},
        "pilot": {"controller_config": dict(PILOT_CONTROLLER_CONFIG),
                  "train_seeds": dict(PILOT_TRAIN_SEEDS),
                  "smoke_seeds": dict(PILOT_SMOKE_SEEDS), "uses_test_for_selection": False},
    }


def graph_fixture():
    pairs = torch.triu_indices(8, 8, 1)
    return Data(num_nodes=8, x=torch.arange(24).reshape(8, 3).float(),
                y=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), orig_id=torch.arange(8),
                edge_index=pairs[:, :20].clone()), pairs


class PrespecifiedPilotRunnerTest(unittest.TestCase):
    def setUp(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="pilot_runner_cpu_", dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.spec = pilot_spec(self.output)
        output_patch = mock.patch.object(runner, "OUTPUT", self.output)
        output_patch.start()
        self.addCleanup(output_patch.stop)

    def test_pilot_provenance_bypass_is_explicit_and_strict_remains_default(self):
        selection = self.spec["controllers"]["sp"]["selection"]
        with mock.patch.object(runner, "file_record", side_effect=AssertionError("validation file")):
            self.assertEqual(runner.operating_point_evidence(
                self.spec, selection, "validation", self.spec["root_seeds"]), selection)
            self.assertEqual(runner.operating_point_evidence(
                self.spec, self.spec["gcn"]["selection"], "independent_validation", []),
                self.spec["gcn"]["selection"])
        for strict in ({}, {"policy": "validation_selected"}):
            with self.subTest(strict=strict), self.assertRaisesRegex(ValueError, "selection basis"):
                runner.operating_point_evidence(strict, selection, "validation", [])
        for change in ({"uses_test_for_selection": True}, {"checkpoint_rule": "best"},
                       {"basis": "validation"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.operating_point_evidence(
                    self.spec, {**selection, **change}, "validation", [])

    def test_pilot_configuration_and_final_epoch_are_not_selectable(self):
        runner.validate_pilot_plan(self.spec)
        for mutate in (
                lambda spec: spec["gcn"]["config"].update(lr=0.02),
                lambda spec: spec["pilot"]["controller_config"].update(controller_epochs=200),
                lambda spec: spec["pilot"]["smoke_seeds"].update(sp=930001),
                lambda spec: spec["root_seeds"].__setitem__(0, 920001)):
            changed = deepcopy(self.spec)
            mutate(changed)
            with self.assertRaises(ValueError):
                runner.validate_pilot_plan(changed)
        args = SimpleNamespace(**PILOT_CONTROLLER_CONFIG, prespecified_pilot=True,
                               seed=PILOT_TRAIN_SEEDS["sp"])
        entry = self.spec["controllers"]["sp"]
        with self.assertRaisesRegex(ValueError, "never best/last"):
            runner.verify_pilot_final(
                {**entry, "checkpoint": str(self.output / "controller_best.pt")}, args,
                {"epoch": 99}, "sp")
        with self.assertRaisesRegex(ValueError, "final epoch"):
            runner.verify_pilot_final(entry, args, {"epoch": 98}, "sp")

    def test_missing_final_checkpoints_allow_preparation_but_block_execution_without_cuda(self):
        graph, _pairs = graph_fixture()
        baseline = SimpleNamespace(num_timesteps=256, parameters=lambda: [])
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "OUTPUT", self.output))
            stack.enter_context(mock.patch.object(runner, "file_record", side_effect=lambda path: {
                "path": str(path), "sha256": "synthetic", "bytes": 0}))
            stack.enter_context(mock.patch.object(runner, "load_args", return_value=SimpleNamespace(
                dataset="cora", diffusion_steps=256)))
            checkpoint = stack.enter_context(mock.patch.object(
                runner, "load_checkpoint", return_value={"model": {}}))
            stack.enter_context(mock.patch.object(runner, "read_reference", return_value=(
                graph, object(), {"full_pair_count": 28})))
            stack.enter_context(mock.patch.object(runner, "build_model", return_value=baseline))
            for name in ("is_available", "device_count", "get_device_name", "synchronize",
                         "manual_seed", "manual_seed_all", "set_device"):
                stack.enter_context(mock.patch(f"torch.cuda.{name}", side_effect=AssertionError("CUDA")))
            manifest = runner.preflight(self.spec)
            self.assertEqual(manifest["status"], "ready_for_controller_training_awaiting_gpu_approval")
            self.assertFalse(manifest["e1_ready"])
            self.assertEqual(set(manifest["pending_controller_artifacts"]), {"sp", "eo"})
            self.assertEqual(len(manifest["blockers"]), 2)
            self.assertTrue(all("training pending" in item for item in manifest["blockers"]))
            self.assertEqual(manifest["fixed_gcn"]["config"], PILOT_GCN_CONFIG)
            checkpoint.assert_called_once_with(self.spec["backbone_checkpoint"])
            with self.assertRaisesRegex(ValueError, "Preflight blockers"):
                runner.execute(self.spec, manifest, stage="smoke")
            self.assertEqual(manifest["actual_counts"]["graphs"], 0)
            self.assertEqual(manifest["actual_counts"]["gcn_fits"], 0)
            self.assertFalse((self.output / "smoke").exists())

    def test_final_audit_rejects_test_use_extra_training_and_changed_assets(self):
        entry = self.spec["controllers"]["sp"]
        args = SimpleNamespace(**PILOT_CONTROLLER_CONFIG, prespecified_pilot=True,
                               seed=PILOT_TRAIN_SEEDS["sp"],
                               controller_pretrained_ckpt=self.spec["backbone_checkpoint"],
                               pilot_backbone_args=self.spec["backbone_args"],
                               pilot_graph=self.spec["graph"])
        audit = {
            "status": "complete", "policy": "prespecified_pilot", "target": "sp",
            "seed": args.seed, "configuration": dict(PILOT_CONTROLLER_CONFIG),
            "completed_epochs": 100, "final_epoch": 99,
            "test_used": False, "real_reference_evaluator_constructed": False,
            "automatic_best_reload": False, "hyperparameter_grid": False,
            "counts": {"optimizer_steps": 100, "replay_sample_calls": 10, "replay_graphs": 20,
                       "auto_export_graphs": 0, "evaluation_graphs": 0, "gcn_fits": 0, "retries": 0},
            "assets": {asset: {"sha256": self.spec[asset]}
                       for asset in ("backbone_checkpoint", "backbone_args", "graph")},
            "backbone_unchanged": True, "trainable_names": sorted(runner.CONTROLLER_KEYS),
            "final_checkpoint": {"sha256": entry["checkpoint"]},
            "controller_diagnostics": {name: {"grad_finite": True, "param_finite": True,
                "param_changed": False, "grad_nonzero_epochs": 0} for name in runner.CONTROLLER_KEYS},
        }
        audit_path = Path(entry["training_audit"])
        audit_path.parent.mkdir(parents=True)
        with mock.patch.object(runner, "file_record", side_effect=lambda path: {"sha256": str(path)}):
            audit_path.write_text(json.dumps(audit))
            self.assertEqual(runner.verify_pilot_final(entry, args, {"epoch": 99}, "sp")["audit"], audit)
            # An unchanged but finite controller is recorded, never replaced by another run.
            for mutate in (
                    lambda item: item.update(test_used=True),
                    lambda item: item["counts"].update(optimizer_steps=101),
                    lambda item: item["counts"].update(auto_export_graphs=1),
                    lambda item: item.update(backbone_unchanged=False),
                    lambda item: item["assets"]["graph"].update(sha256="different graph"),
                    lambda item: item["final_checkpoint"].update(sha256="different controller"),
                    lambda item: item["controller_diagnostics"]["fair_score_eta_raw"].update(grad_finite=False)):
                changed = deepcopy(audit)
                mutate(changed)
                audit_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.verify_pilot_final(entry, args, {"epoch": 99}, "sp")

    def test_smoke_is_three_separate_roots_fits_and_six_real_observations(self):
        graph, pairs = graph_fixture()
        spec = self.spec
        manifest = {
            "status": "ready_awaiting_gpu_approval", "policy": "prespecified_pilot",
            "settings": spec, "blockers": [], "assets": {
                target: {"checkpoint": {"sha256": target}} for target in ("sp", "eo")},
            "actual_counts": {"graphs": 0, "gcn_fits": 0, "gcn_fit_attempts": 0,
                              "observations": 0, "gpu_smoke": 0, "retries": 0},
            "actual_stages": ["cpu_preflight"], "environment": {}, "cost": {},
        }
        parent_manifest = self.output / "manifest.json"
        parent_manifest.write_text("E1 preparation marker\n")
        args_by_path = {spec["backbone_args"]: SimpleNamespace(target="uncontrolled")}
        args_by_path.update({entry["args"]: SimpleNamespace(target=target, fair_score_eo_min_mass=1e-6)
                             for target, entry in spec["controllers"].items()})
        checkpoints = {spec["backbone_checkpoint"]: {"model": {"frozen": torch.tensor([1.])}}}
        checkpoints.update({entry["checkpoint"]: {"target": target}
                            for target, entry in spec["controllers"].items()})
        sampled, fitted, decoders = [], [], []
        sampler = object()

        def build(args, actual_sampler, _backbone, controller=None, *, device):
            self.assertIs(actual_sampler, sampler)
            self.assertEqual(device, "cuda:4")
            self.assertEqual(controller is None, args.target == "uncontrolled")

            def sample(count, *, proxy_observer, proxy_observer_progress):
                self.assertEqual(count, 1)
                self.assertEqual(proxy_observer is None, args.target == "uncontrolled")
                sampled.append(args.target)
                if proxy_observer is not None:
                    same = graph.y[pairs[0]] == graph.y[pairs[1]]
                    for progress in proxy_observer_progress:
                        q = torch.linspace(0.1, 0.9, pairs.shape[1])
                        snapshot = {
                            "target": args.target, "metric": args.target, "q": q,
                            "pair_ids": pairs, "same_mask": same,
                            "pair_batch": torch.zeros(pairs.shape[1], dtype=torch.long),
                            "requested_progress": progress, "loop_index": math.ceil(progress * 256) - 1,
                            "valid_graph": torch.tensor([True]),
                        }
                        if args.target == "eo":
                            snapshot["w"] = torch.linspace(0.2, 0.7, pairs.shape[1])
                        proxy_observer(snapshot)
                return Batch.from_data_list([graph.clone()])
            return SimpleNamespace(sample=sample)

        def fit(generated, config, *, split_seed, fit_seed, device, chunk_size, policy):
            self.assertIs(config, spec["gcn"]["config"])
            self.assertEqual(policy, "prespecified_pilot")
            self.assertEqual(device, "cuda:4")
            self.assertEqual(chunk_size, 5)
            torch.testing.assert_close(generated.x, graph.x)
            fitted.append((split_seed, fit_seed))
            embedding = torch.arange(16).reshape(8, 2).float() / 20
            decoder = EmbeddingDecoder(embedding, chunk_size)
            decoders.append(decoder)
            test_pairs = pairs[:, [0, 3, 20, 22]]
            return {"decoder": decoder, "embedding": embedding, "test_pairs": test_pairs,
                    "test_labels": torch.tensor([1, 1, 0, 0]), "test_scores": decoder(test_pairs),
                    "train_pairs": pairs[:, 4:20], "val_pairs": pairs[:, [1, 2, 21, 23]],
                    "meta": {"gcn_fits": 1, "gcn_best_epoch": 1}}

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "OUTPUT", self.output))
            stack.enter_context(mock.patch.object(runner, "load_args", side_effect=args_by_path.__getitem__))
            stack.enter_context(mock.patch.object(runner, "load_checkpoint", side_effect=checkpoints.__getitem__))
            stack.enter_context(mock.patch.object(runner, "read_reference", return_value=(graph, sampler, {})))
            stack.enter_context(mock.patch.object(runner, "build_model", side_effect=build))
            stack.enter_context(mock.patch.object(runner, "fit_terminal_gcn", side_effect=fit))
            seeds = stack.enter_context(mock.patch.object(runner, "seed_all"))
            stack.enter_context(mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}))
            stack.enter_context(mock.patch("torch.cuda.is_available", return_value=True))
            stack.enter_context(mock.patch("torch.cuda.device_count", return_value=5))
            selected_device = stack.enter_context(mock.patch("torch.cuda.set_device"))
            stack.enter_context(mock.patch("torch.cuda.reset_peak_memory_stats"))
            stack.enter_context(mock.patch("torch.cuda.max_memory_allocated", return_value=0))
            stack.enter_context(mock.patch("torch.cuda.max_memory_reserved", return_value=0))
            stack.enter_context(mock.patch("torch.cuda.get_device_name", return_value="CPU fixture"))
            sync = stack.enter_context(mock.patch("torch.cuda.synchronize"))
            analysis = stack.enter_context(mock.patch.object(metrics, "analyze_snapshot", wraps=metrics.analyze_snapshot))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runner.execute(spec, manifest, stage="smoke")

        roots = [PILOT_SMOKE_SEEDS[target] for target in ("uncontrolled", "sp", "eo")]
        self.assertEqual(sampled, ["uncontrolled", "sp", "eo"])
        selected_device.assert_called_once_with(4)
        self.assertEqual(seeds.call_args_list, [mock.call(root, "cuda:4") for root in roots])
        self.assertEqual(fitted, [(root + 1000000, root + 2000000) for root in roots])
        self.assertFalse(set(roots) & set(spec["root_seeds"]))
        self.assertEqual(sync.call_args_list, [mock.call("cuda:4")] * 3)
        self.assertEqual(analysis.call_count, 6)
        for index, call in enumerate(analysis.call_args_list):
            self.assertIs(call.kwargs["decode_pairs"], decoders[1 + index // 3])
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["actual_counts"]["graphs"], 3)
        self.assertEqual(manifest["actual_counts"]["gcn_fits"], 3)
        self.assertEqual(manifest["actual_counts"]["observations"], 6)
        self.assertEqual(manifest["actual_counts"]["gpu_smoke"], 1)
        self.assertEqual(manifest["cost"]["budget_category"], "preparation_smoke")
        self.assertEqual(parent_manifest.read_text(), "E1 preparation marker\n")
        self.assertFalse((self.output / "artifacts").exists())
        for name, expected in (("terminal_metrics.csv", 3), ("proxy_alignment.csv", 6)):
            with (self.output / "smoke" / name).open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), expected)
        self.assertEqual(len(list((self.output / "smoke/artifacts").glob("*.pt"))), 3)

    def test_smoke_completion_checks_hashes_and_settings_without_quality_threshold(self):
        smoke_dir = self.output / "smoke"
        smoke_dir.mkdir()
        path = smoke_dir / "manifest.json"
        smoke = {"status": "complete", "stage": "smoke", "settings": self.spec,
                 "actual_counts": {"graphs": 3, "observations": 6, "gcn_fit_attempts": 3,
                                   "gcn_fits": 0},
                 "assets": {
                     **{target: {"checkpoint": {"sha256": self.spec["controllers"][target]["checkpoint"]}}
                        for target in ("sp", "eo")},
                     **{asset: {"sha256": self.spec[asset]}
                        for asset in ("backbone_checkpoint", "backbone_args", "graph")}},
                 "metrics": {"auc": float("nan"), "eo": 1.0, "q_g_spearman": -1.0}}
        def record(filename):
            return {"sha256": str(filename)}
        with mock.patch.object(runner, "OUTPUT", self.output), \
             mock.patch.object(runner, "file_record", side_effect=record):
            path.write_text(json.dumps(smoke))
            runner.check_smoke_completed(self.spec)
            for changed in (dict(smoke, status="failed"),
                            dict(smoke, settings={**self.spec, "decode_chunk_size": 10}),
                            dict(smoke, assets={**smoke["assets"], "sp": {"checkpoint": {"sha256": "changed"}}})):
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runner.check_smoke_completed(self.spec)


if __name__ == "__main__":
    unittest.main()
