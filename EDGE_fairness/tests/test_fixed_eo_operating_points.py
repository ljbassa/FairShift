import unittest

from scripts.select_fixed_eo_operating_points import select_operating_points


def row(eta, auc, eo, seed=0, **changes):
    result = {
        "dataset": "cora", "eta": eta, "k": 1, "seed": seed,
        "generate_returncode": 0, "generated_eval_returncode": 0,
        "num_evaluated_graphs": 8, "num_loaded_graphs": 8,
        "lp/auc_mean": auc, "lp/eo_abs_gap_mean": eo,
        "lp/auc_std": 0.02, "lp/eo_abs_gap_std": 0.03,
    }
    result.update(changes)
    return result


def select(rows):
    return select_operating_points(rows, "cora", "/tmp/summary.csv")


class FixedEOOperatingPointsTest(unittest.TestCase):
    def test_auc_threshold_uses_uncontrolled_and_same_run_can_fill_both_roles(self):
        result = select([row(0, .9, .2), row(.1, .95, .1), row(.2, .895, .05)])
        self.assertEqual([item["role"] for item in result],
                         ["uncontrolled", "auc_retained", "fairness_oriented"])
        self.assertEqual([item["eta"] for item in result], [0, .2, .2])

    def test_aggregates_seeds_before_selection_and_reports_seed_std(self):
        result = select([
            row(0, .9, .2, 0), row(0, .92, .2, 1),
            row(.1, .94, .001, 0), row(.1, .84, .19, 1),
            row(.2, .905, .1, 0), row(.2, .915, .1, 1),
        ])
        self.assertEqual(result[1]["eta"], .2)
        self.assertEqual(result[2]["eta"], .1)
        self.assertAlmostEqual(result[2]["auc_seed_mean"], .89)
        self.assertAlmostEqual(result[2]["auc_seed_std"], .05)
        self.assertEqual(result[2]["seeds"], "0;1")
        self.assertEqual(result[2]["num_seeds"], 2)

    def test_excludes_incomplete_failed_undefined_and_unequal_count_candidates(self):
        baseline = [row(0, .9, .2, seed) for seed in (0, 1)]
        cases = [
            [row(.1, .91, .01)],
            [row(.1, .91, .01), row(.1, .91, .01, 1, generated_eval_returncode=1)],
            [row(.1, .91, .01), row(.1, .91, .01, 1, **{"lp/eo_defined_mean": .5})],
            [row(.1, .91, .01), row(.1, .91, .01, 1, num_evaluated_graphs=7)],
            [row(.1, .91, .01), row(.1, float("nan"), .01, 1)],
            [row(.1, .91, .01), row(.1, .91, .01, 1, fair_score_metric="sp")],
            [row(.1, .91, .01), row(.1, .91, .01, 1, **{"aggregate_lp/num_graphs": 7})],
        ]
        for candidates in cases:
            with self.subTest(candidates=candidates):
                result = select(baseline + candidates)
                self.assertEqual(result[1]["status"], "no_qualifying_run")
                self.assertEqual(result[2]["status"], "no_qualifying_run")

    def test_baseline_must_exist_and_be_complete_successful_and_finite(self):
        cases = [
            [row(.1, .9, .1)],
            [row(0, .9, .2), row(.1, .9, .1, 1)],
            [row(0, .9, .2, generate_returncode=1)],
            [row(0, .9, float("nan"))],
            [row(0, .9, .2), row(0, .9, .2, 1, num_evaluated_graphs=7, num_loaded_graphs=7)],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                select(rows)

    def test_duplicates_are_rejected_and_ties_resolve_by_auc_then_eta_k(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select([row(0, .9, .2), row(.1, .9, .1), row(.1, .9, .1)])
        result = select([row(0, .9, .2), row(.3, .91, .1),
                         row(.2, .92, .1), row(.1, .92, .1)])
        self.assertEqual(result[1]["eta"], .1)

    def test_no_qualifying_roles_have_blank_metrics(self):
        result = select([row(0, .9, .2), row(.1, .9, .2)])
        for item in result[1:]:
            self.assertEqual(item["status"], "no_qualifying_run")
            self.assertEqual(item["auc_seed_mean"], "")
            self.assertEqual(item["eo_seed_mean"], "")
        result = select([row(0, .9, .2), row(.1, .88, .1)])
        self.assertEqual(result[1]["status"], "no_qualifying_run")
        self.assertEqual(result[2]["status"], "selected")


if __name__ == "__main__":
    unittest.main()
