"""Tiny CPU calibration tests; no real dataset/controller training is run."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import direct_gap_ablation as ablation
from direct_gap_runtime import attach_fixed_schedule, file_hash
from test_direct_gap_observer import make_model
from test_controller_ablations import MODES, make_controller, make_replay


torch.set_num_threads(2)


def specification(root, variant="T"):
    for name in ("backbone.pt", "args.pickle", "graph.pickle"):
        (root / name).write_bytes(b"tiny test asset, never loaded as an actual checkpoint")
    return {
        "schema_version": 1, "device": "cpu",
        "calibration": {"k0": .4, "variant": variant, "seed": 21, "epochs": 3,
                        "lr": .01, "replay_samples": 1, "replay_refresh": 2,
                        "fairness_weight": 1., "utility_weight": .1, "eta_init": .8,
                        "clip_value": None, "clip_norm": 1.},
        "configurations": [{
            "id": "toy", "dataset": "toy", "target": "sp",
            "backbone_checkpoint": str(root / "backbone.pt"),
            "backbone_args": str(root / "args.pickle"), "graph": str(root / "graph.pickle"),
            "controller": {"kind": "F", "k": .4, "eta": .8, "metric": "sp",
                           "normalization": True, "uses_test_for_selection": False,
                           "configuration_source": "toy only, explicit test settings"},
            "runs": [{"graph_seed": 11, "split_seed": 12, "evaluator_seeds": [13]}],
        }],
    }


class DirectGapAblationTest(unittest.TestCase):
    def test_eta_calibration_preserves_all_native_controller_modes(self):
        for eta_mode, k_mode in MODES:
            for variant in ("T", "tied"):
                with self.subTest(eta_mode=eta_mode, k_mode=k_mode, variant=variant):
                    model = make_controller(eta_mode, k_mode)
                    attach_fixed_schedule(model, 1., [.05] * model.num_timesteps)
                    native_eta = model.fair_score_eta_raw
                    native_k = model.fair_score_k_raw
                    name, parameter = ablation.enable_eta_only(model, eta_init=.05, variant=variant)
                    before = ablation._frozen_digest(model, name)
                    model.fair_score_k_tracking_loss_weight = 0.
                    self.assertEqual(parameter.numel(), model.num_timesteps if variant == "T" else 1)
                    self.assertEqual([key for key, value in model.named_parameters() if value.requires_grad], [name])
                    optimizer = torch.optim.Adam([parameter], lr=.02)
                    initial_eta = model._get_effective_fair_score_eta().detach().clone()
                    for _ in range(2):
                        optimizer.zero_grad()
                        loss, _ = model.compute_fair_controller_loss_from_replay(make_replay())
                        loss.backward()
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        optimizer.step()
                    self.assertEqual(ablation._frozen_digest(model, name), before)
                    self.assertEqual((model.fair_score_eta_mode, model.fair_score_k_mode), (eta_mode, k_mode))
                    self.assertIs(model.fair_score_eta_raw, native_eta)
                    self.assertIs(model.fair_score_k_raw, native_k)
                    if eta_mode == "shared":
                        self.assertFalse(native_eta.requires_grad)
                        torch.testing.assert_close(native_eta, torch.zeros(1), rtol=0, atol=0)
                    eta = model._get_effective_fair_score_eta()
                    self.assertFalse(torch.equal(initial_eta, eta))
                    self.assertEqual(tuple(eta.shape), (model.num_timesteps,))
                    if variant == "tied":
                        torch.testing.assert_close(eta, eta[0].expand_as(eta), rtol=0, atol=0)
                    torch.testing.assert_close(model._get_effective_fair_score_k(), torch.ones(3), rtol=0, atol=0)

    def test_prepare_is_concrete_matched_and_never_trains(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = specification(root)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(spec))
            args = SimpleNamespace(command="prepare-k-ablation", manifest=str(manifest),
                                   output=str(root / "prepared"), allow_calibration=False)
            with patch.object(ablation, "build_observed_model", side_effect=AssertionError("no model in preparation")):
                self.assertEqual(ablation.run_ablation(args), 0)
            prepared = json.loads((root / "prepared/matched_calibration_manifest.json").read_text())
            first, second = prepared["ablation_plan"]
            self.assertEqual((first["fixed_k"], second["fixed_k"]), (.4, 1.))
            self.assertEqual(first["calibration"], second["calibration"])
            self.assertEqual(first["assets"], second["assets"])
            self.assertEqual(second["tracking_loss_weight"], 0)
            self.assertIn("persistent cache retained", second["interpretation"])
            with self.assertRaises(FileExistsError):
                ablation.run_ablation(args)

    def test_calibration_requires_explicit_command_and_flag_before_any_work(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(command="calibrate-k-ablation", manifest="does_not_exist.json",
                                   output=str(Path(temp) / "must_not_exist"), allow_calibration=False)
            with patch.object(ablation, "build_observed_model") as build:
                with self.assertRaisesRegex(ValueError, "--allow-calibration"):
                    ablation.run_ablation(args)
                self.assertFalse(build.called)
            self.assertFalse(Path(args.output).exists())
            args.command, args.allow_calibration = "observe", True
            with self.assertRaisesRegex(ValueError, "explicit command"):
                ablation.run_ablation(args)

    def test_real_replay_objective_only_changes_eta_with_exact_k1_and_tied_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for variant in ("T", "tied"):
                spec = specification(root, variant)
                config, calibration = spec["configurations"][0], spec["calibration"]
                built = []

                def builder(config, *, device):
                    model = make_model(config["target"])
                    model.register_parameter("toy_backbone", torch.nn.Parameter(torch.tensor([2.])))
                    attach_fixed_schedule(model, config["controller"]["k"], [config["controller"]["eta"]] * 4)
                    data = model.initial_graph_sampler.sample(1)
                    data.x = torch.ones(4, 2)
                    built.append(model)
                    return model, data, {"backbone_hash": file_hash(config["backbone_checkpoint"]),
                                         "T": 4, "args": {"fair_label_attr": "y"}}

                checkpoints = []
                for k in (.4, 1.):
                    checkpoint, trace = ablation.calibrate_arm(
                        config, calibration, fixed_k=k, device="cpu", code_revision={"revision": "toy"},
                        model_builder=builder)
                    checkpoints.append(checkpoint)
                    model = built[-1]
                    torch.testing.assert_close(model._get_effective_fair_score_k(), torch.full((4,), k), rtol=0, atol=0)
                    self.assertEqual(model.toy_backbone.item(), 2.)
                    self.assertFalse(model.toy_backbone.requires_grad)
                    self.assertFalse(model.fair_score_k_raw.requires_grad)
                    self.assertEqual(sum(p.requires_grad for p in model.parameters()), 1)
                    self.assertEqual(model.fair_score_k_tracking_loss_weight, 0)
                    self.assertEqual(checkpoint["frozen_state_hash_before"], checkpoint["frozen_state_hash_after"])
                    self.assertEqual(checkpoint["replay_seeds"], [21, 23])
                    self.assertEqual(checkpoint["checkpoint_rule"], "last_epoch_only")
                    self.assertEqual(checkpoint["backbone_args_hash"], file_hash(config["backbone_args"]))
                    self.assertEqual(checkpoint["fair_label_attr"], "y")
                    self.assertEqual(checkpoint["generation_mode"], "not_generated_by_calibration")
                    self.assertEqual(len(trace), calibration["epochs"])
                    self.assertTrue(all(torch.isfinite(torch.tensor(row["loss"])) for row in trace))
                    self.assertTrue(any(abs(eta - .8) > 1e-5 for eta in checkpoint["eta_schedule"]))
                    if variant == "tied":
                        self.assertEqual(len(set(checkpoint["eta_schedule"])), 1)
                self.assertEqual(checkpoints[0]["objective"], checkpoints[1]["objective"])
                self.assertEqual(checkpoints[0]["budget"], checkpoints[1]["budget"])

    def test_ambiguous_or_changed_initialization_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = specification(Path(temp))
            for key, bad in (("k0", 1.), ("variant", "learned-k"), ("eta_init", 0.),
                             ("epochs", 0), ("tracking_loss_weight", .1), ("learned_k", True)):
                with self.subTest(key=key):
                    modified = json.loads(json.dumps(spec))
                    modified["calibration"][key] = bad
                    with self.assertRaises(ValueError):
                        ablation.validate_calibration(modified)
            spec["configurations"][0]["controller"]["kind"] = "T"
            with self.assertRaisesRegex(ValueError, "explicit F"):
                ablation.validate_calibration(spec)

    def test_prepared_asset_changes_are_rejected_before_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = specification(root)
            prepared = ablation.prepare_plan(spec, {"revision": "toy"})
            manifest = root / "prepared.json"
            manifest.write_text(json.dumps(prepared))
            (root / "backbone.pt").write_bytes(b"changed")
            args = SimpleNamespace(command="calibrate-k-ablation", manifest=str(manifest),
                                   output=str(root / "calibrated"), allow_calibration=True)
            with patch.object(ablation, "build_observed_model") as build:
                with self.assertRaisesRegex(ValueError, "asset hashes changed"):
                    ablation.run_ablation(args)
                self.assertFalse(build.called)

    def test_shared_source_configuration_groups_replicates_but_separates_k_arms(self):
        """A shared hyperparameter ID must not average k0 and k1 together."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = specification(root)
            first = spec["configurations"][0]
            first["id"], first["configuration_id"] = "controller_seed_101", "shared_settings"
            second = json.loads(json.dumps(first))
            second["id"] = "controller_seed_102"
            second["runs"] = [{"graph_seed": 21, "split_seed": 22, "evaluator_seeds": [23]}]
            spec["configurations"].append(second)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(spec))
            args = SimpleNamespace(command="calibrate-k-ablation", manifest=str(manifest),
                                   output=str(root / "calibrated"), allow_calibration=True)

            def completed_stub(config, calibration, *, fixed_k, device, code_revision):
                # Seed identity is supplied by this fixture, not a real calibration.
                return {"variant": "T", "fixed_k": fixed_k,
                        "calibration_seed": int(config["id"].rsplit("_", 1)[1])}, []

            with patch.object(ablation, "calibrate_arm", side_effect=completed_stub) as calibrate, \
                    patch.object(ablation, "build_observed_model", side_effect=AssertionError("no real model")):
                self.assertEqual(ablation.run_ablation(args), 0)
            self.assertEqual(calibrate.call_count, 4)
            observed = json.loads((root / "calibrated/observe_manifest.json").read_text())
            entries = observed["configurations"]
            self.assertEqual(len({entry["id"] for entry in entries}), 4)
            groups = {}
            for entry in entries:
                groups.setdefault(entry["configuration_id"], []).append(entry)
                self.assertEqual(entry["ablation"]["source_configuration_id"], "shared_settings")
            self.assertEqual(set(groups), {"shared_settings__k0", "shared_settings__k1"})
            for configuration_id, expected_k in (("shared_settings__k0", .4), ("shared_settings__k1", 1.)):
                group = groups[configuration_id]
                self.assertEqual(len(group), 2)
                checkpoints = [torch.load(entry["controller"]["checkpoint"], weights_only=True) for entry in group]
                self.assertEqual({checkpoint["calibration_seed"] for checkpoint in checkpoints}, {101, 102})
                self.assertTrue(all(checkpoint["fixed_k"] == expected_k for checkpoint in checkpoints))
                self.assertEqual({entry["runs"][0]["graph_seed"] for entry in group}, {11, 21})


if __name__ == "__main__":
    unittest.main()
