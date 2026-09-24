"""CPU protocol checks for the prespecified pilot's actual terminal fit."""

from copy import deepcopy
import unittest
from unittest import mock

import numpy as np
import torch
from torch_geometric.data import Data

import proxy_minimal_gcn as fixed


class PrespecifiedPilotGCNTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        pairs = torch.triu_indices(10, 10, 1)[:, :20]
        self.data = Data(
            edge_index=torch.cat((pairs, pairs.flip(0)), dim=1),
            x=torch.arange(30).reshape(10, 3).float() / 30,
            y=torch.arange(10) % 2, num_nodes=10)

    def fit(self):
        return fixed.fit_terminal_gcn(
            self.data, fixed.PILOT_GCN_CONFIG, split_seed=12, fit_seed=15,
            device="cpu", chunk_size=3, policy="prespecified_pilot")

    def test_actual_fit_uses_prespecified_optimizer_encoder_and_val_patience(self):
        fixed.seed_all(12)
        split = fixed.samplepy_prepare_for_gae(self.data)
        val_pairs = split["val_mask"].nonzero().T
        val_labels = (split["A_full"][val_pairs[0], val_pairs[1]] != 0).long().numpy()

        def validation_auc(labels, _scores):
            np.testing.assert_array_equal(labels, val_labels)
            return 0.75

        with mock.patch.object(fixed, "safe_auc", side_effect=validation_auc) as auc, \
             mock.patch.object(fixed, "PrespecifiedPilotGAE", wraps=fixed.PrespecifiedPilotGAE) as model, \
             mock.patch.object(torch.optim, "Adam", wraps=torch.optim.Adam) as adam, \
             mock.patch.object(fixed.F, "dropout", wraps=fixed.F.dropout) as dropout, \
             mock.patch("evaluate_generated_graphs.samplepy_config_list", side_effect=AssertionError("grid")), \
             mock.patch("evaluate_generated_graphs.samplepy_device", side_effect=AssertionError("implicit GPU")):
            result = self.fit()

        model.assert_called_once_with(3, 1, 128, 0.1)
        self.assertEqual(adam.call_args.kwargs, {"lr": 0.01, "weight_decay": 0.0})
        self.assertEqual(result["meta"]["gcn_epochs_run"], 6)
        self.assertEqual(result["meta"]["gcn_best_epoch"], 1)
        self.assertEqual(result["meta"]["gcn_config"]["max_epochs"], 1000)
        self.assertEqual(auc.call_count, 6)
        self.assertEqual(result["meta"]["gcn_config"], fixed.PILOT_GCN_CONFIG)
        self.assertEqual(result["meta"]["gcn_policy"], "prespecified_pilot")
        self.assertEqual(result["meta"]["gcn_dropout_application"], "input_features_training_only")
        self.assertEqual(len(dropout.call_args_list), 13)
        self.assertEqual(sum(call.kwargs["training"] for call in dropout.call_args_list), 6)
        self.assertTrue(all(call.kwargs["p"] == 0.1 for call in dropout.call_args_list))
        for call in dropout.call_args_list:
            torch.testing.assert_close(call.args[0], self.data.x)
        self.assertFalse(result["embedding"].requires_grad)
        before = result["embedding"].clone()
        for _ in range(3):
            torch.testing.assert_close(result["decoder"](result["test_pairs"]), result["test_scores"])
        torch.testing.assert_close(result["embedding"], before)

    def test_test_labels_cannot_change_epoch_or_terminal_embedding(self):
        fixed.seed_all(12)
        original = fixed.samplepy_prepare_for_gae(self.data)
        changed = deepcopy(original)
        test = changed["test_mask"]
        changed["A_full"][test] = (changed["A_full"][test] == 0).float()
        with mock.patch.object(fixed, "samplepy_prepare_for_gae", return_value=original), \
             mock.patch.object(fixed, "safe_auc", return_value=0.75):
            first = self.fit()
        with mock.patch.object(fixed, "samplepy_prepare_for_gae", return_value=changed), \
             mock.patch.object(fixed, "safe_auc", return_value=0.75):
            second = self.fit()
        self.assertFalse(torch.equal(first["test_labels"], second["test_labels"]))
        self.assertEqual(first["meta"]["gcn_best_epoch"], second["meta"]["gcn_best_epoch"])
        torch.testing.assert_close(first["embedding"], second["embedding"], rtol=0, atol=0)
        torch.testing.assert_close(first["test_scores"], second["test_scores"], rtol=0, atol=0)

    def test_pilot_rejects_any_unprescribed_setting_before_split_or_fit(self):
        with mock.patch.object(fixed, "samplepy_prepare_for_gae", side_effect=AssertionError("split")):
            for key, value in (("lr", 0.001), ("dropout", 0.0), ("patience", 10),
                               ("hidden_size", 64), ("num_layers", 2), ("max_epochs", 10)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    fixed.fit_terminal_gcn(
                        self.data, {**fixed.PILOT_GCN_CONFIG, key: value},
                        split_seed=12, fit_seed=15, device="cpu", policy="prespecified_pilot")

    def test_strict_encoder_remains_legacy_and_pilot_dropout_is_training_only(self):
        adjacency = torch.eye(10)
        features = torch.ones(10, 3)
        pilot = fixed.PrespecifiedPilotGAE(3, 1, 128, 0.1)
        legacy = fixed.SamplePyGAE(3, 1, 128, 0.1)
        legacy.load_state_dict(pilot.state_dict())
        pilot.eval()
        legacy.eval()
        torch.testing.assert_close(pilot(adjacency, features), legacy(adjacency, features))
        pilot.train()
        legacy.train()
        with mock.patch.object(fixed.F, "dropout", wraps=fixed.F.dropout) as dropout:
            legacy(adjacency, features)
            dropout.assert_not_called()
            pilot(adjacency, features)
            dropout.assert_called_once_with(features, p=0.1, training=True)


if __name__ == "__main__":
    unittest.main()
