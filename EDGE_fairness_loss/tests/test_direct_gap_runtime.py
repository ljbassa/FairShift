"""Contract tests for the fixed-controller and existing-evaluator adapters."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import direct_gap_runtime as runtime
from test_controller_ablations import MODES, make_controller

torch.set_num_threads(1)


class ControllerContractTest(unittest.TestCase):
    def checkpoint(self):
        return {
            "format": "fairshift_fixed_k_eta_v1", "backbone_hash": "backbone",
            "dataset": "cora", "num_timesteps": 3, "fixed_k": .5,
            "variant": "T", "trainable_parameters": ["eta_t"],
            "tracking_loss_weight": 0, "learned_k": False,
            "uses_test_for_selection": False, "calibration_seed": 41,
            "checkpoint_rule": "last", "objective": "existing_guided_gap",
            "budget": {"epochs": 4}, "code_revision": "revision",
            "eta_schedule": [.01, .02, .03], "metric": "sp",
            "normalization": True,
            "eo_min_mass": 1e-6, "fair_label_attr": "y",
            "backbone_args_hash": "args", "reference_graph_hash": "reference",
            "feature_hash": "features", "group_hash": "groups",
        }

    def validate(self, checkpoint):
        return runtime.validate_controller(checkpoint, backbone_hash="backbone", timesteps=3, dataset="cora")

    def test_legacy_schedule_cannot_be_relabelled_by_constant_raw_k(self):
        legacy = {"controller": {"fair_score_k_raw": torch.zeros(3),
                                  "fair_score_eta_raw": torch.zeros(3)}}
        with self.assertRaisesRegex(ValueError, "legacy learned-k"):
            self.validate(legacy)

    def test_valid_fixed_k_checkpoint_preserves_exact_schedule_and_provenance(self):
        checkpoint = self.checkpoint()
        result = self.validate(checkpoint)
        self.assertEqual(result["fixed_k"], checkpoint["fixed_k"])
        self.assertEqual(result["fair_label_attr"], "y")
        self.assertEqual(result["eo_min_mass"], 1e-6)
        torch.testing.assert_close(torch.tensor(result["eta_schedule"]), torch.tensor(checkpoint["eta_schedule"]))

    def test_rejects_wrong_backbone_tracking_and_false_tied_schedule(self):
        for changes in ({"backbone_hash": "other"}, {"tracking_loss_weight": .01},
                        {"variant": "tied", "trainable_parameters": ["eta"]},
                        {"uses_test_for_selection": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate({**self.checkpoint(), **changes})

    def test_exact_k_one_retains_cache_and_does_not_rewrite_legacy_tensors(self):
        model = torch.nn.Module()
        model.device, model.num_timesteps = "cpu", 3
        model.fair_score_k_raw = torch.nn.Parameter(torch.tensor([.1, .2, .3]))
        model.fair_score_eta_raw = torch.nn.Parameter(torch.tensor([.4, .5, .6]))
        model._fair_score_q = torch.tensor([.2, .3])
        before = deepcopy(model.state_dict())
        original_cache = model._fair_score_q
        runtime.attach_fixed_schedule(model, 1., [.01, .02, .03])
        torch.testing.assert_close(model._get_effective_fair_score_k(), torch.ones(3), rtol=0, atol=0)
        torch.testing.assert_close(model._get_effective_fair_score_eta(t_graph=torch.tensor([2, 0])),
                                   torch.tensor([.03, .01]))
        self.assertIs(model._fair_score_q, original_cache)
        torch.testing.assert_close(model.fair_score_k_raw, before["fair_score_k_raw"])
        torch.testing.assert_close(model.fair_score_eta_raw, before["fair_score_eta_raw"])
        self.assertFalse(model.fair_score_k_raw.requires_grad)
        self.assertFalse(model.fair_score_eta_raw.requires_grad)

    def test_fixed_adapter_preserves_native_parameter_shapes_and_modes(self):
        for eta_mode, k_mode in MODES:
            with self.subTest(eta_mode=eta_mode, k_mode=k_mode):
                model = make_controller(eta_mode, k_mode)
                before = deepcopy(model.state_dict())
                native_eta = model.fair_score_eta_raw
                native_k = model.fair_score_k_raw
                runtime.attach_fixed_schedule(model, 1., [.01, .02, .03])
                self.assertEqual(model.fair_score_eta_mode, eta_mode)
                self.assertEqual(model.fair_score_k_mode, k_mode)
                self.assertIs(model.fair_score_eta_raw, native_eta)
                self.assertIs(model.fair_score_k_raw, native_k)
                self.assertFalse(native_eta.requires_grad)
                for key, value in before.items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                for query in (torch.tensor(1), torch.tensor([2, 0]), torch.tensor([[2, 1], [0, 2]])):
                    torch.testing.assert_close(model._get_effective_fair_score_k(t_graph=query),
                                               torch.ones(query.shape), rtol=0, atol=0)
                    torch.testing.assert_close(model._get_effective_fair_score_eta(t_graph=query),
                                               torch.tensor([.01, .02, .03])[query], rtol=0, atol=0)

    def test_observe_rejects_training_before_mutating_native_state(self):
        model = make_controller("shared", "fixed_one")
        before = deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "does not calibrate"):
            runtime.attach_fixed_schedule(model, 1., [.01] * 3, train_eta=True)
        self.assertTrue(model.fair_score_eta_raw.requires_grad)
        self.assertFalse(hasattr(model, "_direct_gap_eta"))
        self.assertEqual(set(before), set(model.state_dict()))
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_backbone_loader_accepts_native_controller_keys_only(self):
        import networkx as nx

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = nx.Graph([(0, 1), (2, 3)])
            with (root / "graph.pkl").open("wb") as stream:
                pickle.dump(reference, stream)
            data = types.SimpleNamespace(x=torch.ones(4, 2), y=torch.tensor([0, 0, 1, 1]))
            data_utils = types.ModuleType("datasets.data_utils")
            data_utils.preprocess = mock.Mock(return_value=data)
            data_utils.EmpiricalEmptyGraphGenerator = mock.Mock(return_value=object())
            evaluator = types.ModuleType("evaluate_generated_graphs")
            evaluator.get_local_attr_vector = lambda value, attr: getattr(value, attr)
            evaluator.get_lp_group_vector = lambda value, preferred_attr: getattr(value, preferred_attr)
            model_module = types.ModuleType("model")
            model_module.get_model = lambda args, sampler: make_controller(
                args.fair_score_eta_mode, args.fair_score_k_mode)
            config = {
                "dataset": "toy", "target": "sp", "backbone_args": str(root / "args.pkl"),
                "graph": str(root / "graph.pkl"), "backbone_checkpoint": str(root / "backbone.pt"),
                "controller": {"kind": "F", "k": 1., "eta": .02, "metric": "sp",
                               "normalization": True, "uses_test_for_selection": False,
                               "configuration_source": "explicit compatibility fixture"},
            }
            with mock.patch.dict(sys.modules, {"datasets.data_utils": data_utils,
                                               "evaluate_generated_graphs": evaluator,
                                               "model": model_module}):
                for eta_mode, k_mode in MODES:
                    with self.subTest(eta_mode=eta_mode, k_mode=k_mode):
                        args = types.SimpleNamespace(
                            dataset="toy", degree=True, num_node_feat=2, max_degree=1,
                            empty_graph_sampler="empirical", augmented_features=False, diffusion_steps=3,
                            fair_score_eta_mode=eta_mode, fair_score_k_mode=k_mode,
                        )
                        with (root / "args.pkl").open("wb") as stream:
                            pickle.dump(args, stream)
                        native = make_controller(eta_mode, k_mode)
                        state = native.state_dict()
                        torch.save({"model": state}, root / "backbone.pt")
                        model, _, actual = runtime.build_observed_model(config, device="cpu")
                        self.assertEqual((model.fair_score_eta_mode, model.fair_score_k_mode),
                                         (eta_mode, k_mode))
                        self.assertEqual(actual["args"]["fair_score_eta_mode"], eta_mode)
                        self.assertEqual(actual["args"]["fair_score_k_mode"], k_mode)
                        torch.testing.assert_close(model._denoise_fn.weight, native._denoise_fn.weight)
                        torch.testing.assert_close(model._get_effective_fair_score_k(), torch.ones(3))
                        # Only absent controller tensors are permitted missing.
                        for invalid in ({key: value for key, value in state.items() if key != "_denoise_fn.weight"},
                                        {**state, "unexpected_backbone_tensor": torch.ones(1)}):
                            torch.save({"model": invalid}, root / "backbone.pt")
                            with self.assertRaisesRegex(ValueError, "Strict backbone"):
                                runtime.build_observed_model(config, device="cpu")


class ExistingEvaluatorAdapterTest(unittest.TestCase):
    def test_existing_split_and_validation_selection_reused_with_one_test_prediction(self):
        """Forward actual masks unchanged; select by validation, not test/proxy."""
        old = types.ModuleType("evaluate_generated_graphs")
        original_device = lambda: torch.device("cpu")
        old.samplepy_device = original_device
        data = types.SimpleNamespace(num_nodes=4, x=torch.arange(8).reshape(4, 2).float(),
                                     y=torch.tensor([0, 0, 1, 1]), sens=None,
                                     edge_index=torch.tensor([[0, 1], [1, 2]]))
        train = torch.zeros((4, 4), dtype=torch.bool)
        val = train.clone()
        test = train.clone()
        train[0, 1], val[1, 2], test[0, 3] = True, True, True
        labels = torch.zeros((4, 4))
        labels[0, 1], labels[1, 2] = 1, 1
        split = {"A_train": torch.eye(4), "A_full": labels,
                 "train_mask": train, "val_mask": val, "test_mask": test}
        old.ensure_features = mock.Mock(side_effect=lambda value: value)
        old.samplepy_prepare_for_gae = mock.Mock(return_value=split)
        old.get_lp_group_vector = mock.Mock(return_value=data.y)
        old.samplepy_preprocess = mock.Mock(return_value=(split["A_train"], data.x, labels))
        configs = [{"lr": .1}, {"lr": .01}, {"lr": .001}]
        old.samplepy_config_list = mock.Mock(return_value=configs)
        models = [torch.nn.Linear(2, 1) for _ in configs]
        old.samplepy_fit_trial = mock.Mock(side_effect=[
            (.7, 999., 999., models[0], {"best_val_auc": .7}),
            (.9, -999., -999., models[1], {"best_val_auc": .9}),
            (.8, 0., 0., models[2], {"best_val_auc": .8}),
        ])
        old.samplepy_predict = mock.Mock(return_value=(.4, 0., 0.,
                    {"scores": np.array([.3]), "labels": np.array([0])}))
        old.unique_undirected_edge_index = mock.Mock(return_value=data.edge_index)
        seeding = types.ModuleType("proxy_minimal_gcn")
        seeding.seed_all = mock.Mock()
        with mock.patch.dict(sys.modules, {"evaluate_generated_graphs": old,
                                          "proxy_minimal_gcn": seeding}):
            result = runtime.fit_existing_evaluator(data, split_seed=17, evaluator_seed=19, device="cpu")
        self.assertEqual(seeding.seed_all.call_args_list, [mock.call(17, "cpu"), mock.call(19, "cpu")])
        old.samplepy_prepare_for_gae.assert_called_once_with(data)
        self.assertEqual(old.samplepy_fit_trial.call_count, len(configs))
        for call, config in zip(old.samplepy_fit_trial.call_args_list, configs):
            self.assertIs(call.args[4], train)
            self.assertIs(call.args[5], val)
            self.assertEqual(call.kwargs, config)
        old.samplepy_predict.assert_called_once()
        self.assertIs(old.samplepy_predict.call_args.args[4], test)
        self.assertIs(old.samplepy_predict.call_args.args[5], models[1])
        self.assertIs(old.samplepy_device, original_device)
        torch.testing.assert_close(result["pair_ids"], test.nonzero().t())
        self.assertEqual(result["selected_model_meta"]["best_val_auc"], .9)
        self.assertEqual(result["auc"], .4)
        self.assertEqual((result["split_seed"], result["evaluator_seed"]), (17, 19))

    def test_device_override_is_restored_after_evaluator_exception(self):
        old = types.ModuleType("evaluate_generated_graphs")
        previous = lambda: "original"
        old.samplepy_device = previous
        with mock.patch.dict(sys.modules, {"evaluate_generated_graphs": old}):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with runtime.evaluator_device("cpu"):
                    self.assertEqual(old.samplepy_device(), torch.device("cpu"))
                    raise RuntimeError("failure")
        self.assertIs(old.samplepy_device, previous)


class OfflineArtifactJoinTest(unittest.TestCase):
    def test_rejects_content_mismatch_even_when_artifact_hashes_are_current(self):
        import run_direct_gap as runner
        for mutation, expected_error in (
            ("graph", "different generated graph"),
            ("order", "node-order"),
            ("features", "feature content"),
            ("groups", "group source"),
            ("checkpoint", "provenance mismatch"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = types.SimpleNamespace(
                    num_nodes=4, x=torch.arange(8).reshape(4, 2).float(),
                    y=torch.tensor([0, 0, 1, 1]), orig_id=torch.arange(4),
                    edge_index=torch.tensor([[0, 1], [1, 2]]),
                )
                identity = {"graph_hash": runtime.graph_hash(data),
                            "node_order_hash": runtime.tensor_hash(data.orig_id),
                            "controller_hash": "controller-a"}
                trajectory = {"data": data, "snapshots": [], "provenance": deepcopy(identity)}
                evaluation = {"generated_positive_pairs": data.edge_index.clone(),
                              "provenance": {**identity, "evaluator_seed": 3},
                              "feature_hash": runtime.tensor_hash(data.x),
                              "node_groups": data.y.clone(), "group_hash": runtime.tensor_hash(data.y),
                              "group_source": "generated_graph.y"}
                if mutation == "graph":
                    evaluation["generated_positive_pairs"] = torch.tensor([[0, 1], [2, 2]])
                elif mutation == "order":
                    data.orig_id = data.orig_id.flip(0)
                elif mutation == "features":
                    data.x = data.x + 1
                elif mutation == "groups":
                    evaluation["node_groups"] = torch.tensor([0, 1, 0, 1])
                    evaluation["group_hash"] = runtime.tensor_hash(evaluation["node_groups"])
                else:
                    evaluation["provenance"]["controller_hash"] = "controller-b"
                artifacts = {}
                for key, value in (("trajectory", trajectory), ("evaluation", evaluation)):
                    path = root / f"{key}.pt"
                    torch.save(value, path)
                    artifacts[key] = {"path": path.name, "sha256": runtime.file_hash(path)}
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({"schema_version": 1, "artifacts": [artifacts]}))
                args = types.SimpleNamespace(manifest=str(manifest), output=str(root / "output"))
                with mock.patch.object(runner, "code_identity", return_value={}), \
                        mock.patch.object(runner, "write_results"), \
                        self.assertRaisesRegex(ValueError, expected_error):
                    runner.analyze_saved(args)


if __name__ == "__main__":
    unittest.main()
