"""Small CPU protocol tests, never an experiment/configuration search."""

import unittest
from unittest import mock

import numpy as np
import torch
from torch_geometric.data import Data

from evaluate_generated_graphs import samplepy_prepare_for_gae
from proxy_minimal_gcn import (EmbeddingDecoder, InvalidTerminalFit, fit_terminal_gcn,
                               seed_all, validate_config)


class FixedGCNTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        pairs = torch.triu_indices(10, 10, 1)[:, :20]
        self.data = Data(edge_index=torch.cat((pairs, pairs.flip(0)), dim=1),
                         x=torch.arange(30).reshape(10, 3).float() / 30,
                         y=torch.arange(10) % 2, num_nodes=10)
        # This deliberately tiny synthetic-test setting is never an E1 default.
        self.config = dict(lr=0.01, num_layers=1, hidden_size=4, dropout=0.0,
                           max_epochs=3, patience=2, batch_size=16, min_delta=0.0,
                           weight_decay=0.0, selection_metric="validation_auc")

    def test_split_is_generated_80_10_10_and_train_only_adjacency(self):
        seed_all(12)
        split = samplepy_prepare_for_gae(self.data)
        self.assertEqual([int(split[k]) for k in ("num_train_pos", "num_val_pos", "num_test_pos")], [16, 2, 2])
        train, val, test = (split[k] for k in ("train_mask", "val_mask", "test_mask"))
        self.assertFalse((train & val).any())
        self.assertFalse((train & test).any())
        self.assertFalse((val & test).any())
        self.assertEqual(int((split["A_train"][val | test] != 0).sum()), 0)
        self.assertTrue((split["A_train"][train] > 0).all())
        self.assertTrue((split["A_train"].diag() > 0).all())
        torch.testing.assert_close(split["A_train"], split["A_train"].T)

    def test_one_fit_uses_fixed_values_and_never_legacy_grid(self):
        import proxy_minimal_gcn as fixed
        with mock.patch("evaluate_generated_graphs.samplepy_config_list", side_effect=AssertionError("grid")), \
             mock.patch("evaluate_generated_graphs.samplepy_train_and_eval", side_effect=AssertionError("legacy")), \
             mock.patch.object(fixed, "SamplePyGAE", wraps=fixed.SamplePyGAE) as factory, \
             mock.patch.object(fixed, "safe_auc", wraps=fixed.safe_auc) as auc:
            result = fit_terminal_gcn(self.data, self.config, split_seed=12,
                                      fit_seed=15, device="cpu", chunk_size=2)
        factory.assert_called_once_with(3, 1, 4, 0.0)
        self.assertEqual(result["meta"]["gcn_fits"], 1)
        self.assertLessEqual(result["meta"]["gcn_epochs_run"], 3)
        self.assertEqual(auc.call_count, result["meta"]["gcn_epochs_run"])
        self.assertEqual(result["embedding"].device.type, "cpu")
        self.assertFalse(result["embedding"].requires_grad)
        before = result["embedding"].clone()
        for _ in range(3):
            torch.testing.assert_close(result["decoder"](result["test_pairs"]), result["test_scores"])
        torch.testing.assert_close(before, result["embedding"])

    def test_chunk_decode_matches_pairwise_reference(self):
        z = torch.arange(28).reshape(7, 4).float() / 30
        pairs = torch.triu_indices(7, 7, 1)
        actual = EmbeddingDecoder(z, chunk_size=3)(pairs)
        expected = torch.sigmoid(z @ z.T)[pairs[0], pairs[1]]
        torch.testing.assert_close(actual, expected)

    def test_config_missing_or_test_selected_is_rejected(self):
        for config in (None, {}, {**self.config, "selection_metric": "test_auc"}):
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_invalid_graph_does_not_invent_a_fit(self):
        self.data.edge_index = torch.tensor([[0, 1], [1, 0]])
        with self.assertRaises(InvalidTerminalFit) as caught:
            fit_terminal_gcn(self.data, self.config, split_seed=12, fit_seed=15, device="cpu")
        self.assertEqual(caught.exception.fits, 0)


if __name__ == "__main__":
    unittest.main()
