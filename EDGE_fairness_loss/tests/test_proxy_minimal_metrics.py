import unittest

import numpy as np
import torch

from proxy_minimal_metrics import (
    analyze_snapshot,
    decode_chunked,
    safe_correlation,
    summarize_rows,
    terminal_metrics,
)


class ProxyMinimalMetricsTest(unittest.TestCase):
    def setUp(self):
        self.pairs = torch.triu_indices(4, 4, offset=1)
        self.groups = torch.tensor([0, 0, 1, 1])
        self.same = self.groups[self.pairs[0]] == self.groups[self.pairs[1]]
        self.q = torch.tensor([0.8, 0.3, 0.4, 0.6, 0.2, 0.7], dtype=torch.float64)
        self.w = torch.tensor([0.6, 0.3, 0.8, 0.2, 0.9, 0.4], dtype=torch.float64)
        self.g = torch.tensor([0.9, 0.2, 0.5, 0.4, 0.1, 0.6], dtype=torch.float64)
        self.positive_index = torch.tensor([0, 2, 3, 5])
        self.positive_pairs = self.pairs[:, self.positive_index]
        self.test_index = torch.tensor([0, 1, 2, 4, 5])
        self.test_pairs = self.pairs[:, self.test_index]
        self.test_labels = np.array([1, 0, 1, 0, 1])
        self.lookup = {tuple(pair): i for i, pair in enumerate(self.pairs.t().tolist())}
        self.decode_sizes = []

    def decode(self, pairs):
        self.assertEqual(pairs.device.type, "cpu")
        self.assertEqual(pairs.shape[0], 2)
        self.decode_sizes.append(pairs.shape[1])
        return self.g[[self.lookup[tuple(sorted(pair))] for pair in pairs.t().tolist()]]

    def snapshot(self, target):
        snapshot = {"metric": target, "pair_ids": self.pairs,
                    "q": self.q, "same_mask": self.same,
                    "requested_progress": 0.25, "loop_index": 249,
                    "step_number": 250, "diffusion_t": 750,
                    "progress": 0.25, "total_steps": 1000}
        if target == "eo":
            snapshot["w"] = self.w
        return snapshot

    def analyze(self, target, **overrides):
        snapshot = self.snapshot(target)
        snapshot.update(overrides)
        return analyze_snapshot(snapshot, decode_pairs=self.decode,
                                generated_positive_pairs=self.positive_pairs,
                                test_pairs=self.test_pairs, test_labels=self.test_labels,
                                num_nodes=4, chunk_size=2)

    def test_sp_telescoping_on_full_cache_and_test_support(self):
        result = self.analyze("sp")
        self.assertAlmostEqual(result["d"], 0.375)
        self.assertAlmostEqual(result["R"], 0.075)
        self.assertEqual(result["C"], 0.0)
        self.assertAlmostEqual(result["S"], 1.0 / 30.0)
        self.assertAlmostEqual(result["terminal_gap"], 29.0 / 60.0)
        self.assertTrue(result["identity_holds"])
        self.assertEqual(result["q_g_pair_count"], 5)
        self.assertEqual(result["d_same_count"], 2)
        self.assertEqual(result["d_different_count"], 4)
        self.assertLessEqual(max(self.decode_sizes), 2)
        self.assertEqual(sum(self.decode_sizes), 11)

    def test_eo_uses_separate_weights_and_generated_positive_conditioning(self):
        result = self.analyze("eo")
        q_v = (0.6 * 0.8 + 0.4 * 0.7) - (0.3 * 0.3 + 0.8 * 0.4 + 0.2 * 0.6 + 0.9 * 0.2) / 2.2
        g_v = (0.6 * 0.9 + 0.4 * 0.6) - (0.3 * 0.2 + 0.8 * 0.5 + 0.2 * 0.4 + 0.9 * 0.1) / 2.2
        self.assertAlmostEqual(result["d"], q_v)
        self.assertAlmostEqual(result["R"], g_v - q_v)
        self.assertAlmostEqual(result["C"], 0.3 - g_v)
        self.assertAlmostEqual(result["S"], -0.05)
        self.assertAlmostEqual(result["terminal_gap"], 0.25)
        self.assertTrue(result["identity_holds"])
        self.assertEqual(result["Q_y_g_same_count"], 2)
        self.assertEqual(result["Q_y_g_different_count"], 2)
        self.assertEqual(result["q_g_pair_count"], 3)
        positive_test_index = self.test_index[self.test_labels.astype(bool)]
        expected = safe_correlation(self.q[positive_test_index], self.g[positive_test_index])
        self.assertAlmostEqual(result["q_g_spearman"], expected["value"])

    def test_terminal_metrics_use_test_positives_for_soft_eo(self):
        result = terminal_metrics(self.g[self.test_index], self.test_labels,
                                  self.test_pairs, self.groups, self.positive_pairs, 4,
                                  generated_positive_scores=self.g[self.positive_index])
        self.assertAlmostEqual(result["eo"], 0.25)
        self.assertAlmostEqual(result["sp"], 29.0 / 60.0)
        self.assertAlmostEqual(result["auc"], 1.0)
        self.assertAlmostEqual(result["density"], 4.0 / 6.0)
        self.assertEqual(result["test_positive_different_count"], 1)
        self.assertEqual(result["generated_positive_different_count"], 2)
        self.assertAlmostEqual(result["test_positive_different_score_mean"], 0.5)
        self.assertAlmostEqual(result["generated_positive_different_score_mean"], 0.45)

    def test_missing_group_and_negligible_eo_mass_are_nan(self):
        missing = self.analyze("eo", same_mask=torch.ones(6, dtype=torch.bool))
        self.assertTrue(np.isnan(missing["d"]))
        self.assertEqual(missing["d_valid_count"], 0)
        self.assertFalse(missing["identity_holds"])
        tiny = self.w.clone()
        tiny[self.same] = 1e-12
        result = self.analyze("eo", w=tiny)
        self.assertTrue(np.isnan(result["d"]))
        self.assertEqual(result["d_valid_count"], 0)
        self.assertEqual(result["d_same_count"], 2)
        self.assertEqual(result["d_same_finite_count"], 2)

    def test_nonfinite_values_do_not_silently_change_moment_support(self):
        bad_q = self.q.clone()
        bad_q[0] = float("nan")
        result = self.analyze("sp", q=bad_q)
        self.assertTrue(np.isnan(result["d"]))
        self.assertEqual(result["d_same_count"], 2)
        self.assertEqual(result["d_same_finite_count"], 1)
        self.assertEqual(result["q_g_pair_count"], 4)
        self.assertEqual(result["q_g_total_pair_count"], 5)

    def test_constant_correlation_preserves_count_and_nan(self):
        result = safe_correlation([1.0, 1.0, float("nan")], [0.1, 0.9, 0.5])
        self.assertTrue(np.isnan(result["value"]))
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["total_pair_count"], 3)
        self.assertEqual(result["valid_count"], 0)
        single = safe_correlation([1.0], [0.2])
        self.assertTrue(np.isnan(single["value"]))
        self.assertEqual(single["pair_count"], 1)

    def test_support_lookup_handles_reordered_and_reversed_pairs(self):
        order = torch.tensor([4, 0, 5, 2, 1, 3])
        actual = self.analyze("eo", pair_ids=self.pairs[:, order].flip(0),
                              q=self.q[order], w=self.w[order], same_mask=self.same[order])
        expected = self.analyze("eo")
        for key in ("d", "R", "C", "S", "q_g_spearman"):
            self.assertAlmostEqual(actual[key], expected[key])

    def test_rejects_test_labels_from_a_different_graph_and_missing_cache_pairs(self):
        args = dict(decode_pairs=self.decode, generated_positive_pairs=self.positive_pairs,
                    test_pairs=self.test_pairs, num_nodes=4, chunk_size=2)
        with self.assertRaisesRegex(ValueError, "completed generated graph"):
            analyze_snapshot(self.snapshot("sp"), test_labels=1 - self.test_labels, **args)
        snapshot = self.snapshot("sp")
        snapshot.update(pair_ids=self.pairs[:, 1:], q=self.q[1:], same_mask=self.same[1:])
        with self.assertRaisesRegex(ValueError, "absent from production cache"):
            analyze_snapshot(snapshot, test_labels=self.test_labels, **args)

    def test_observer_production_proxy_comparison(self):
        result = self.analyze("sp", production_proxy=torch.tensor([0.375], dtype=torch.float64),
                              valid_graph=torch.tensor([True]))
        self.assertAlmostEqual(result["production_proxy_residual"], 0.0)
        self.assertEqual(result["production_proxy_valid_count"], 1)
        invalid = self.analyze("sp", production_proxy=torch.tensor([0.0]), valid_graph=torch.tensor([False]))
        self.assertTrue(np.isnan(invalid["production_proxy"]))
        self.assertEqual(invalid["production_proxy_valid_count"], 0)

    def test_graph_level_summaries_keep_targets_and_timesteps_separate(self):
        rows = []
        for target in ("sp", "eo"):
            for progress in (0.25, 0.5, 0.9):
                for seed in range(3):
                    rows.append({"target": target, "requested_progress": progress, "root_seed": seed,
                                 "d": float(seed), "terminal_gap": float(-seed if target == "eo" else seed),
                                 "R_abs": float(seed), "C_abs": 0.0, "S_abs": 0.1,
                                 "q_g_spearman": 0.2, "identity_residual": 0.0})
        terminal = [{"arm": arm, "root_seed": seed, "eo_abs": float(seed)}
                    for arm in ("uncontrolled", "sp", "eo") for seed in range(3)]
        result = summarize_rows(terminal, rows)
        self.assertEqual(len(result["terminal"]), 3)
        self.assertEqual(len(result["alignment"]), 6)
        for row in result["alignment"]:
            self.assertEqual(row["graph_count"], 3)
            self.assertEqual(row["signed_proxy_terminal_pearson_graph_count"], 3)
            self.assertAlmostEqual(row["signed_proxy_terminal_pearson"], -1.0 if row["target"] == "eo" else 1.0)
            self.assertAlmostEqual(row["R_abs_mean"], 1.0)
            self.assertAlmostEqual(row["R_abs_sem"], 1.0 / np.sqrt(3))
        with self.assertRaisesRegex(ValueError, "Duplicate root_seed"):
            summarize_rows(terminal, rows + [rows[0]])

    def test_decoder_reuses_fixed_embedding_and_only_decodes_requested_chunks(self):
        embeddings = torch.tensor([[1.0, 0.0], [0.1, 0.2], [0.0, 1.0], [0.3, 0.2]])
        original = embeddings.clone()
        calls = []

        def decode(pairs):
            calls.append(pairs.shape[1])
            return (embeddings[pairs[0]] * embeddings[pairs[1]]).sum(-1).sigmoid()

        actual = decode_chunked(decode, self.pairs, chunk_size=2)
        expected = (embeddings[self.pairs[0]] * embeddings[self.pairs[1]]).sum(-1).sigmoid().numpy()
        np.testing.assert_allclose(actual, expected)
        torch.testing.assert_close(embeddings, original)
        self.assertEqual(calls, [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
