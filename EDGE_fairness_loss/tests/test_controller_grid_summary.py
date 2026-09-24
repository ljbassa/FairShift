import ast
import csv
import io
import json
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import summarize_controller_grid as summary


class ReadControllerMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "controller_metrics.jsonl"

    def assert_warning_location(self, warning, line_number):
        self.assertTrue(issubclass(warning.category, RuntimeWarning))
        self.assertRegex(
            str(warning.message),
            rf"controller_metrics\.jsonl(?::|.*line\s+){line_number}\b",
        )

    def test_missing_empty_and_blank_files_return_empty_without_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(summary.read_jsonl_last(self.path), {})
            for content in ("", "\n   \n\t\n"):
                with self.subTest(content=content):
                    self.path.write_text(content, encoding="utf-8")
                    self.assertEqual(summary.read_jsonl_last(self.path), {})
        self.assertEqual(caught, [])

    def test_valid_records_ignore_blanks_and_return_last_record(self):
        expected = {"epoch": 2, "loss": 0.25, "fair_score_metric": "eo"}
        self.path.write_text(
            '\n{"epoch": 1, "loss": 0.5}\n  \n' + json.dumps(expected),
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(summary.read_jsonl_last(self.path), expected)
        self.assertEqual(caught, [])

    def test_malformed_middle_line_does_not_hide_later_valid_record(self):
        for damaged in ("progress: finished epoch 2", '\x00\x00{"epoch": 2}'):
            with self.subTest(damaged=damaged):
                self.path.write_text(
                    '{"epoch": 1}\n\n' + damaged + '\n{"epoch": 3}\n',
                    encoding="utf-8",
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.assertEqual(summary.read_jsonl_last(self.path), {"epoch": 3})
                self.assertEqual(len(caught), 1)
                self.assert_warning_location(caught[0], 3)

    def test_truncated_final_record_preserves_last_complete_epoch(self):
        expected = {"epoch": 19, "loss": 0.125}
        self.path.write_text(
            json.dumps(expected) + '\n{"epoch": 20, "loss":',
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(summary.read_jsonl_last(self.path), expected)
        self.assertEqual(len(caught), 1)
        self.assert_warning_location(caught[0], 2)

    def test_non_object_json_values_preserve_last_record_and_warn(self):
        expected = {"epoch": 7, "loss": 0.2}
        for value in (None, [], [1, 2], "status", 12, 0.5, True):
            with self.subTest(value=value):
                self.path.write_text(
                    json.dumps(expected) + "\n" + json.dumps(value) + "\n",
                    encoding="utf-8",
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.assertEqual(summary.read_jsonl_last(self.path), expected)
                self.assertEqual(len(caught), 1)
                self.assert_warning_location(caught[0], 2)

    def test_all_invalid_records_return_empty_with_each_location(self):
        self.path.write_text("invalid\nnull\n[]\n{\n", encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(summary.read_jsonl_last(self.path), {})
        self.assertEqual(len(caught), 4)
        for line_number, warning in enumerate(caught, start=1):
            self.assert_warning_location(warning, line_number)


class ControllerGridSummaryTest(unittest.TestCase):
    def test_eo_csv_survives_damaged_metrics_and_missing_loss_sort_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "controller" / "eo"
            run_metrics = {
                "grid_a_missing": "broken metrics\nnull\n",
                "grid_b_truncated": (
                    '{"epoch": 9, "loss": 0.2, "fair_score_metric": "eo"}\n'
                    '{"epoch": 10, "loss":'
                ),
                "grid_c_healthy": (
                    '{"epoch": 10, "loss": 0.1, "fair_score_metric": "eo"}\n'
                ),
            }
            for name, metrics in run_metrics.items():
                run_dir = root / name
                samples = run_dir / "generated_samples"
                samples.mkdir(parents=True)
                (run_dir / "controller_metrics.jsonl").write_text(metrics, encoding="utf-8")
                gap = 0.1 if name == "grid_c_healthy" else 0.2
                (samples / "lp_summary.csv").write_text(
                    "lp/auc_mean,lp/auc_std,lp/eo_abs_gap_mean,lp/eo_abs_gap_std,"
                    "lp/eo_defined_mean,aggregate_lp/auc,aggregate_lp/eo_abs_gap,"
                    "lp/score_sp_gap_mean,aggregate_lp/score_sp_abs_gap\n"
                    f"0.8,0.02,{gap},0.01,1.0,0.82,{gap},0.3,0.4\n",
                    encoding="utf-8",
                )

            out_csv = root / "grid_summary.csv"
            argv = [
                "summarize_controller_grid.py",
                "--controller_root", str(root),
                "--fair_score_metric", "eo",
                "--prefix", "grid_",
                "--sort_by", "lp/eo_abs_gap_mean",
                "--out_csv", str(out_csv),
            ]
            output = io.StringIO()
            with mock.patch("sys.argv", argv), \
                    mock.patch.object(summary, "load_args_from_checkpoint", return_value={
                        "fair_score_metric": "eo",
                    }), \
                    warnings.catch_warnings(record=True) as caught, \
                    redirect_stdout(output):
                warnings.simplefilter("always")
                summary.main()

            self.assertEqual(len(caught), 3)
            with out_csv.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["run_name"] for row in rows],
                ["grid_c_healthy", "grid_b_truncated", "grid_a_missing"],
            )
            self.assertEqual([row["fair_score_metric"] for row in rows], ["eo"] * 3)
            self.assertEqual(rows[1]["last_epoch"], "9")
            self.assertEqual(rows[1]["loss"], "0.2")
            self.assertEqual(rows[2]["last_epoch"], "")
            self.assertEqual(rows[2]["loss"], "")
            self.assertEqual(rows[2]["lp/eo_abs_gap_mean"], "0.2")
            self.assertEqual(rows[2]["generated/lp/auc_mean"], "0.8")
            previews = [
                ast.literal_eval(line)
                for line in output.getvalue().splitlines()
                if line.startswith("{")
            ]
            self.assertEqual(len(previews), 3)
            self.assertEqual(previews[0]["lp/auc_mean"], 0.8)
            self.assertEqual(previews[0]["lp/auc_std"], 0.02)
            self.assertEqual(previews[0]["lp/eo_abs_gap_mean"], 0.1)
            self.assertEqual(previews[0]["lp/eo_abs_gap_std"], 0.01)
            self.assertEqual(previews[0]["lp/eo_defined_mean"], 1.0)
            self.assertEqual(previews[0]["generated/aggregate_lp/auc"], 0.82)
            self.assertEqual(previews[0]["generated/aggregate_lp/eo_abs_gap"], 0.1)
            self.assertNotIn("lp/score_sp_gap_mean", previews[0])
            self.assertNotIn("generated/aggregate_lp/score_sp_abs_gap", previews[0])


if __name__ == "__main__":
    unittest.main()
