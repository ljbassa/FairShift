"""CPU failures distinguish numerical corruption from structural undefined AUC."""

from copy import deepcopy
import unittest
from unittest import mock

import torch
from torch_geometric.data import Data

import proxy_minimal_gcn as fixed


class NumericalGuardTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        pairs = torch.triu_indices(10, 10, 1)[:, :20]
        self.data = Data(
            edge_index=torch.cat((pairs, pairs.flip(0)), dim=1),
            x=torch.arange(30).reshape(10, 3).float() / 30,
            y=torch.arange(10) % 2, num_nodes=10)
        # Synthetic CPU protocol check; this does not alter the pilot config.
        self.config = dict(lr=0.01, num_layers=1, hidden_size=4, dropout=0.0,
                           max_epochs=1, patience=1, batch_size=16, min_delta=0.0,
                           weight_decay=0.0, selection_metric="validation_auc")

    def fit(self):
        return fixed.fit_terminal_gcn(
            self.data, self.config, split_seed=12, fit_seed=15, device="cpu")

    def test_nonfinite_loss_stops_before_backward_or_optimizer_update(self):
        with mock.patch.object(fixed.F, "binary_cross_entropy_with_logits",
                               return_value=torch.tensor(float("nan"), requires_grad=True)), \
             mock.patch.object(torch.optim.Adam, "step") as step, \
             self.assertRaisesRegex(RuntimeError, "training loss"):
            self.fit()
        step.assert_not_called()

    def test_nonfinite_gradient_stops_before_optimizer_update(self):
        original = fixed.SamplePyGAE

        def corrupt_gradient(*args):
            model = original(*args)
            next(model.parameters()).register_hook(lambda grad: grad * float("nan"))
            return model

        with mock.patch.object(fixed, "SamplePyGAE", side_effect=corrupt_gradient), \
             mock.patch.object(torch.optim.Adam, "step") as step, \
             self.assertRaisesRegex(RuntimeError, "gradient"):
            self.fit()
        step.assert_not_called()

    def test_nonfinite_validation_embedding_stops(self):
        original = fixed.SamplePyGAE

        class CorruptValidation(original):
            def forward(self, adjacency, features):
                embedding = super().forward(adjacency, features)
                return embedding if self.training else embedding * float("nan")

        with mock.patch.object(fixed, "SamplePyGAE", CorruptValidation), \
             self.assertRaisesRegex(RuntimeError, "validation embedding"):
            self.fit()

    def test_nonfinite_validation_scores_stop(self):
        with mock.patch.object(fixed.EmbeddingDecoder, "__call__",
                               side_effect=lambda pairs: torch.full((pairs.shape[1],), float("nan"))), \
             self.assertRaisesRegex(RuntimeError, "validation scores"):
            self.fit()

    def test_nonfinite_auc_with_two_classes_stops(self):
        with mock.patch.object(fixed, "safe_auc", return_value=float("nan")), \
             self.assertRaisesRegex(RuntimeError, "validation AUC"):
            self.fit()

    def test_nonfinite_terminal_embedding_stops(self):
        original = fixed.SamplePyGAE

        class CorruptTerminal(original):
            calls = 0

            def forward(self, adjacency, features):
                embedding = super().forward(adjacency, features)
                self.calls += 1
                return embedding * float("nan") if self.calls == 3 else embedding

        with mock.patch.object(fixed, "SamplePyGAE", CorruptTerminal), \
             self.assertRaisesRegex(RuntimeError, "terminal embedding"):
            self.fit()

    def test_empty_or_single_class_validation_remains_structurally_invalid(self):
        fixed.seed_all(12)
        original = fixed.samplepy_prepare_for_gae(self.data)
        for empty in (True, False):
            split = deepcopy(original)
            if empty:
                split["val_mask"].zero_()
            else:
                split["A_full"][split["val_mask"]] = 1
            with self.subTest(empty=empty), \
                 mock.patch.object(fixed, "samplepy_prepare_for_gae", return_value=split), \
                 self.assertRaises(fixed.InvalidTerminalFit) as caught:
                self.fit()
            self.assertEqual(caught.exception.fits, 1)


if __name__ == "__main__":
    unittest.main()
