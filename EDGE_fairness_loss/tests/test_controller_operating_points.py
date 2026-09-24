import csv
import tempfile
import unittest
import warnings
from pathlib import Path

from scripts import select_controller_operating_points as selector


def point(name, auc, eo, **extra):
    return dict({"run_name": name, selector.AUC: auc, selector.EO: eo,
                 selector.EO_DEFINED: 1.0}, **extra)


class ControllerOperatingPointsTest(unittest.TestCase):
    def test_auc_floor_uses_uncontrolled_instead_of_grid_maximum(self):
        rows = [point("grid_max", .96, .08), point("retained", .895, .04),
                point("fairness", .86, .02)]
        selected = selector.select_points(rows, point("baseline", .90, .10), "cora")
        self.assertEqual([row["run_name"] for row in selected], ["baseline", "retained", "fairness"])
        self.assertAlmostEqual(selected[1]["auc_drop_from_uncontrolled"], .005)
        self.assertAlmostEqual(selected[1]["eo_reduction_from_uncontrolled"], .06)

    def test_both_roles_are_explicit_when_same_candidate_wins(self):
        selected = selector.select_points([point("winner", .895, .02)],
                                          point("baseline", .9, .1), "citeseer")
        self.assertEqual([row["selection"] for row in selected],
                         ["uncontrolled", "auc_retained", "fairness_oriented"])
        self.assertEqual([row["run_name"] for row in selected[1:]], ["winner", "winner"])

    def test_no_qualifying_run_does_not_fabricate_metrics(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selected = selector.select_points([point("worse_eo", .95, .11), point("equal_eo", .94, .1)],
                                              point("baseline", .9, .1), "cora")
        self.assertEqual(len(caught), 2)
        for row in selected[1:]:
            self.assertEqual(row["status"], "no_qualifying_run")
            self.assertEqual(row[selector.AUC], "")
            self.assertEqual(row[selector.EO], "")
            self.assertEqual(row["run_name"], "")

    def test_fairness_auc_floor_is_optional_and_relative_to_baseline(self):
        rows = [point("retained", .90, .04), point("moderate", .875, .03), point("low_auc", .8, .01)]
        selected = selector.select_points(rows, point("baseline", .9, .1), "cora",
                                          fairness_max_auc_drop=.03)
        self.assertEqual(selected[2]["run_name"], "moderate")

    def test_nonfinite_or_undefined_candidates_are_excluded(self):
        rows = [point("nan_auc", float("nan"), .01), point("infinite_eo", .99, float("inf")),
                point("partial_eo", .99, .001), point("valid", .90, .05)]
        rows[2][selector.EO_DEFINED] = .875
        with self.assertWarnsRegex(RuntimeWarning, "Excluded 3 rows"):
            selected = selector.select_points(rows, point("baseline", .9, .1), "cora")
        self.assertEqual([row["run_name"] for row in selected[1:]], ["valid", "valid"])
        with self.assertRaisesRegex(ValueError, "Uncontrolled baseline"):
            selector.select_points(rows, rows[2], "cora")

    def test_lower_eo_then_higher_auc_then_name_break_ties(self):
        rows = [point("high_auc", .99, .04), point("low_eo", .90, .03),
                point("z_tie", .91, .03), point("a_tie", .91, .03)]
        selected = selector.select_points(rows, point("baseline", .9, .1), "cora")
        self.assertEqual([row["run_name"] for row in selected[1:]], ["a_tie", "a_tie"])

    def test_cli_preserves_configuration_uncertainty_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid, baseline, output = [root / name for name in ("grid.csv", "baseline.csv", "selected.csv")]
            selector.write_csv(grid, [point("candidate", .91, .03, fair_score_eta="500",
                                            **{"lp/auc_std": .02, "lp/eo_abs_gap_std": .01})])
            selector.write_csv(baseline, [point("baseline", .9, .1)])
            selector.main(["--summary_csv", str(grid), "--uncontrolled_summary_csv", str(baseline),
                           "--dataset", "cora", "--out_csv", str(output)])
            with output.open(newline="") as stream:
                selected = list(csv.DictReader(stream))
            self.assertEqual(len(selected), 3)
            self.assertEqual(selected[1]["fair_score_eta"], "500")
            self.assertEqual(selected[1]["lp/auc_std"], "0.02")
            self.assertEqual(selected[1]["summary_csv"], str(grid.resolve()))
            self.assertEqual(selected[1]["uncontrolled_summary_csv"], str(baseline.resolve()))

    def test_invalid_auc_drop_is_rejected(self):
        for value in (-.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                selector.select_points([], point("baseline", .9, .1), "cora", auc_max_drop=value)


if __name__ == "__main__":
    unittest.main()
