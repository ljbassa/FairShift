"""Small CPU-only diagnostics: no backbone, generation, or evaluator fitting."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from direct_gap_analysis import analyze_snapshot, safe_correlation, summarize_records, write_analysis


class DirectGapAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.pairs = np.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=np.int64)
        self.q = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.7])
        self.groups = np.array([0, 0, 1, 1])
        self.meta = dict(graph_id="g0", graph_hash="graph-sha", backbone_hash="backbone-sha",
                         controller_hash="controller-sha", node_order_hash="nodes-sha",
                         pair_mapping="unordered_i_lt_j", batch_id=0, dataset="cora",
                         configuration_id="sp_k05", controller_seed=11, graph_seed=22,
                         protocol="generated_test", feature_source="identity",
                         group_source="existing_node_metadata")
        self.snapshot = dict(pair_ids=self.pairs.copy(), q=self.q.copy(),
                             provenance=self.meta.copy(), phase="post", score_name="q_final",
                             snapshot_id="final", progress=1.0, chunk_index=0,
                             visited_mask=np.array([True, True, False, True, False, True]))
        # P contains both positive and negative pairs, including both groups.
        self.p_indices = np.array([5, 2, 0, 3])
        self.evaluation = dict(pair_ids=self.pairs[:, self.p_indices].copy(),
                               scores=np.array([0.6, 0.5, 0.8, 0.1]), labels=np.array([1, 1, 0, 0]),
                               node_groups=self.groups.copy(), num_nodes=4,
                               generated_positive_pairs=self.pairs[:, [2, 5]].copy(),
                               provenance={**self.meta, "evaluator_seed": 33})

    def analyze(self, snapshot=None, evaluation=None, **kwargs):
        return analyze_snapshot(self.snapshot if snapshot is None else snapshot,
                                self.evaluation if evaluation is None else evaluation, **kwargs)

    def test_signed_sp_and_telescoping_counts_and_units(self):
        sp, eo = self.analyze()
        # a = (.9+.7)/2 - (.1+.2+.3+.4)/4 = .55
        # b = (.9+.7)/2 - (.2+.3)/2 = .55; c = (.8+.6)/2 - (.5+.1)/2 = .4
        self.assertAlmostEqual(sp["a"], 0.55)
        self.assertAlmostEqual(sp["b"], 0.55)
        self.assertAlmostEqual(sp["c"], 0.4)
        self.assertAlmostEqual(sp["total_difference"], sp["support_difference"] + sp["score_difference"])
        self.assertAlmostEqual(sp["c_pp"], 40.0)
        self.assertEqual(sp["a_same_count"], 2)
        self.assertEqual(sp["a_different_count"], 4)
        self.assertAlmostEqual(sp["a_same_sum"], 1.6)
        self.assertEqual(sp["c_same_count"], 2)  # Includes negative (0,1).
        self.assertEqual(sp["cache_unvisited_count"], 2)
        self.assertEqual(sp["cache_coverage"], 1.0)
        self.assertAlmostEqual(sp["gnn_eo"], 0.1)
        self.assertEqual(eo["diagnostic_status"], "unavailable")
        self.assertTrue(np.isnan(eo["a"]))
        self.assertAlmostEqual(eo["d"], 0.1)

    def test_signed_gap_is_not_absolute_gap(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["q"] = 1 - snapshot["q"]
        row = self.analyze(snapshot=snapshot)[0]
        self.assertAlmostEqual(row["a"], -0.55)
        self.assertAlmostEqual(row["a_abs"], 0.55)
        self.assertAlmostEqual(row["total_difference"], -0.95)
        self.assertEqual(row["a_downstream_sign_agreement"], 0.0)

    def test_same_support_has_zero_support_difference(self):
        evaluation = copy.deepcopy(self.evaluation)
        evaluation.update(pair_ids=self.pairs.copy(), scores=np.linspace(0.1, 0.6, 6),
                          labels=np.array([0, 0, 1, 0, 0, 1]))
        row = self.analyze(evaluation=evaluation)[0]
        self.assertEqual(row["support_difference"], 0.0)

    def test_matching_scores_have_zero_score_difference(self):
        evaluation = copy.deepcopy(self.evaluation)
        evaluation["scores"] = self.q[self.p_indices]
        row = self.analyze(evaluation=evaluation)[0]
        self.assertEqual(row["score_difference"], 0.0)

    def test_pair_id_join_survives_independent_reordering(self):
        expected = self.analyze()[0]
        snapshot, evaluation = copy.deepcopy(self.snapshot), copy.deepcopy(self.evaluation)
        order = np.array([3, 0, 2, 5, 1, 4])
        snapshot["pair_ids"] = snapshot["pair_ids"][:, order]
        snapshot["q"] = snapshot["q"][order]
        snapshot["visited_mask"] = snapshot["visited_mask"][order]
        p_order = np.array([2, 3, 1, 0])
        evaluation["pair_ids"] = evaluation["pair_ids"][:, p_order]
        evaluation["scores"] = evaluation["scores"][p_order]
        evaluation["labels"] = evaluation["labels"][p_order]
        actual = self.analyze(snapshot, evaluation)[0]
        for key in ("a", "b", "c", "support_difference", "score_difference", "auc"):
            self.assertAlmostEqual(actual[key], expected[key])

    def test_input_arrays_are_not_mutated(self):
        before = copy.deepcopy(self.snapshot)
        self.analyze()
        for key in ("pair_ids", "q", "visited_mask"):
            np.testing.assert_array_equal(self.snapshot[key], before[key])

    def test_inactive_and_unvisited_scores_are_in_full_cache_gap(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["active_pair_indices"] = np.array([0])
        snapshot["visited_mask"][:] = False
        snapshot["visited_mask"][0] = True
        row = self.analyze(snapshot)[0]
        self.assertEqual(row["a_same_count"] + row["a_different_count"], 6)
        self.assertAlmostEqual(row["a"], 0.55)
        self.assertEqual(row["cache_unvisited_count"], 5)

    def test_pre_post_and_expected_phase_validation(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot.update(phase="pre", score_name="q_bar", progress=0.9, snapshot_id="pre90")
        self.assertEqual(self.analyze(snapshot)[0]["phase"], "pre")
        with self.assertRaisesRegex(ValueError, "phase"):
            self.analyze(snapshot, provenance={"snapshot_phase": "post"})
        snapshot["score_name"] = "q_final"
        with self.assertRaisesRegex(ValueError, "phase/score_name"):
            self.analyze(snapshot)
        snapshot.update(phase="post", progress=.9)
        with self.assertRaisesRegex(ValueError, "progress=1"):
            self.analyze(snapshot)
        with self.assertRaisesRegex(ValueError, "evaluator_seed"):
            self.analyze(provenance={"evaluator_seed": 99})

    def test_graph_checkpoint_node_mapping_and_phase_mismatches_refused(self):
        for key in ("graph_id", "graph_hash", "backbone_hash", "controller_hash", "node_order_hash", "pair_mapping"):
            with self.subTest(key=key):
                evaluation = copy.deepcopy(self.evaluation)
                evaluation["provenance"][key] = "wrong"
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    self.analyze(evaluation=evaluation)
        evaluation = copy.deepcopy(self.evaluation)
        evaluation["provenance"]["snapshot_phase"] = "pre"
        with self.assertRaisesRegex(ValueError, "phase"):
            self.analyze(evaluation=evaluation)
        del evaluation["provenance"]["graph_hash"]
        with self.assertRaisesRegex(ValueError, "Missing independently bound provenance"):
            self.analyze(evaluation=evaluation)

    def test_bad_pair_conventions_duplicates_and_selfloops_rejected(self):
        for kind in ("reversed", "duplicate", "selfloop", "noninteger", "out_of_range"):
            with self.subTest(kind=kind):
                snapshot = copy.deepcopy(self.snapshot)
                if kind == "reversed":
                    snapshot["pair_ids"] = snapshot["pair_ids"][::-1]
                elif kind == "duplicate":
                    snapshot["pair_ids"][:, 1] = snapshot["pair_ids"][:, 0]
                elif kind == "selfloop":
                    snapshot["pair_ids"][1, 0] = 0
                elif kind == "noninteger":
                    snapshot["pair_ids"] = snapshot["pair_ids"].astype(float)
                else:
                    snapshot["pair_ids"][1, 0] = 4
                with self.assertRaises(ValueError):
                    self.analyze(snapshot)

    def test_cross_batch_and_wrong_group_mask_refused(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["node_batch"] = np.array([0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "cross-batch"):
            self.analyze(snapshot)
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["pair_batch"] = np.array([0, 0, 0, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "different graph batch"):
            self.analyze(snapshot)
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["same_mask"] = np.ones(6, dtype=bool)
        with self.assertRaisesRegex(ValueError, "same_mask"):
            self.analyze(snapshot)

    def test_missing_pair_and_reference_labels_refused(self):
        snapshot = copy.deepcopy(self.snapshot)
        for key in ("q", "visited_mask"):
            snapshot[key] = snapshot[key][:-1]
        snapshot["pair_ids"] = snapshot["pair_ids"][:, :-1]
        with self.assertRaisesRegex(ValueError, "missing from the cache"):
            self.analyze(snapshot)
        evaluation = copy.deepcopy(self.evaluation)
        evaluation["labels"][2] = 1
        with self.assertRaisesRegex(ValueError, "completed generated graph"):
            self.analyze(evaluation=evaluation)

    def test_empty_group_and_nan_keep_explicit_invalid_reasons(self):
        evaluation = copy.deepcopy(self.evaluation)
        evaluation["node_groups"][:] = 0
        row = self.analyze(evaluation=evaluation)[0]
        self.assertTrue(np.isnan(row["a"]))
        self.assertIn("different:empty_group", row["a_invalid_reason"])
        self.assertFalse(row["identity_valid"])
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["q"][0] = np.nan
        row = self.analyze(snapshot)[0]
        self.assertTrue(np.isnan(row["a"]))
        self.assertIn("nonfinite_score", row["a_invalid_reason"])
        self.assertTrue(np.isnan(row["a_same_sum"]))
        self.assertTrue(np.isnan(row["a_pp"]))

    def test_eo_weighted_positive_telescoping(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["w"] = np.array([0.2, 0.9, 0.1, 0.7, 0.3, 0.8])
        row = self.analyze(snapshot)[1]
        expected_a = (0.9 * 0.2 + 0.7 * 0.8) - (0.1 * 0.9 + 0.2 * 0.1 + 0.3 * 0.7 + 0.4 * 0.3) / 2.0
        expected_b = (0.9 * 0.2 + 0.7 * 0.8) - (0.2 * 0.1 + 0.3 * 0.7) / 0.8
        self.assertAlmostEqual(row["a"], expected_a)
        self.assertAlmostEqual(row["b"], expected_b)
        self.assertAlmostEqual(row["c"], 0.5)
        self.assertAlmostEqual(row["d"], 0.1)
        self.assertAlmostEqual(row["a"] - row["d"], row["support_difference"] +
                               row["conditioning_difference"] + row["score_difference"])
        self.assertEqual(row["c_same_count"], 1)
        self.assertEqual(row["c_different_count"], 1)
        self.assertTrue(row["identity_valid"])

    def test_eo_zero_negative_and_nonfinite_weights(self):
        for bad in (0.0, -1.0, np.nan):
            snapshot = copy.deepcopy(self.snapshot)
            snapshot["w"] = np.full(6, bad)
            row = self.analyze(snapshot)[1]
            self.assertTrue(np.isnan(row["a"]))
            self.assertTrue(row["a_invalid_reason"])
            self.assertFalse(row["identity_valid"])

    def test_constant_small_nan_correlations_have_reasons(self):
        for x, y, reason in (([1], [2], "insufficient_finite_pairs"),
                             ([1, 1], [2, 3], "constant_input"),
                             ([np.nan, 1], [2, 3], "insufficient_finite_pairs")):
            result = safe_correlation(x, y)
            self.assertTrue(np.isnan(result["value"]))
            self.assertEqual(result["invalid_reason"], reason)
        self.assertAlmostEqual(safe_correlation([3, 1, 2], [6, 2, 4], "spearman")["value"], 1.0)
        self.assertAlmostEqual(safe_correlation([1, 2, 2], [4, 8, 8], "spearman")["value"], 1.0)

    def _rows(self):
        rows = []
        for config, controller, graph, seed, a, b, c in (
                ("A", 1, 1, 1, .1, .2, .3), ("A", 1, 1, 2, .1, .2, .5),
                ("A", 1, 2, 1, .3, .4, .6), ("B", 2, 3, 1, .5, .6, .8)):
            row = self.analyze()[0]
            row.update(configuration_id=config, controller_seed=controller, graph_seed=graph,
                       graph_id=f"g{graph}", graph_hash=f"hash{graph}", evaluator_seed=seed,
                       a=a, b=b, c=c, downstream_gap=c)
            rows.append(row)
        return rows

    def test_evaluator_repeats_do_not_inflate_graph_or_configuration_count(self):
        graph, configuration = summarize_records(self._rows())
        self.assertEqual(graph["analysis_level"], "graph")
        self.assertEqual(graph["unit_count"], 3)
        self.assertEqual(graph["record_count"], 4)
        self.assertEqual(graph["evaluator_fit_count"], 4)
        self.assertEqual(configuration["unit_count"], 2)
        self.assertAlmostEqual(configuration["a_mean"], .35)
        self.assertAlmostEqual(configuration["c_mean"], .65)

    def test_phases_progress_dataset_protocol_are_separate(self):
        base = self.analyze()[0]
        rows = [base]
        for changed in (dict(phase="pre", score_name="q_bar", progress=.9),
                        dict(phase="pre", score_name="q_bar", progress=.5),
                        dict(dataset="citeseer"), dict(protocol="other_generated_test")):
            rows.append({**base, **changed})
        summary = summarize_records(rows)
        self.assertEqual(len(summary), 10)
        self.assertTrue(all(row["unit_count"] == 1 for row in summary))
        self.assertTrue(all(row["signed_a_downstream_pearson_invalid_reason"] == "insufficient_finite_pairs"
                            for row in summary))

    def test_duplicate_snapshots_cannot_be_independent_graph_samples(self):
        row = self.analyze()[0]
        with self.assertRaisesRegex(ValueError, "Duplicate graph/evaluator"):
            summarize_records([row, {**row, "snapshot_id": "another"}])
        with self.assertRaisesRegex(ValueError, "same snapshot"):
            summarize_records([row, {**row, "evaluator_seed": 44, "snapshot_id": "another"}])

    def test_nearzero_sign_mask_is_prespecified(self):
        rows = self._rows()
        rows[0]["a"] = rows[1]["a"] = 1e-8
        graph = summarize_records(rows)[0]
        self.assertEqual(graph["a_sign_valid_count"], 2)
        self.assertEqual(graph["a_sign_excluded_count"], 1)
        rows[0]["sign_threshold"] = .01
        with self.assertRaisesRegex(ValueError, "threshold"):
            summarize_records(rows)

    def test_invalid_repeat_not_silently_dropped(self):
        rows = self._rows()
        rows[0]["a"] = np.nan
        graph = summarize_records(rows)[0]
        self.assertEqual(graph["a_valid_count"], 2)
        self.assertEqual(graph["signed_a_downstream_pearson_excluded_nonfinite_count"], 1)

    def test_csv_metadata_plot_artifacts_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            result = write_analysis(self.analyze(), directory)
            self.assertEqual(len(result["summary"]), 4)  # SP + unavailable EO, two levels.
            self.assertEqual(len(result["plot_paths"]), 24)  # 3 comparisons x 2 formats x 2 levels x 2 metrics.
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in result["plot_paths"]))
            metadata = json.loads((Path(directory) / "analysis_metadata.json").read_text())
            self.assertFalse(metadata["full_cache_gnn_evaluation"])
            self.assertIn("a_same_sum", (Path(directory) / "gap_records.csv").read_text())
            self.assertIn("insufficient_finite_pairs", (Path(directory) / "gap_summary.csv").read_text())
            with self.assertRaises(FileExistsError):
                write_analysis(self.analyze(), directory)


if __name__ == "__main__":
    unittest.main()
