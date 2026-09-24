import argparse
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evaluate
from fair_grid_eval_generated_graphs import build_generate_args


class StopBeforeLoadingData(Exception):
    pass


class FixedGridArgsTest(unittest.TestCase):
    def resolve_saved_args(self, *, metric, eta, normalized, saved_apply):
        """Exercise real CLI parsing and saved-argument overrides without sampling."""
        with tempfile.TemporaryDirectory() as run_dir:
            saved = argparse.Namespace(
                fair_score_sp=False,
                fair_score_metric="sp",
                fair_score_apply_sample=saved_apply,
                fair_score_guidance_normalize=not normalized,
            )
            with (Path(run_dir) / "args.pickle").open("wb") as stream:
                pickle.dump(saved, stream)
            grid_args = argparse.Namespace(
                dataset="cora",
                run_name="test_run",
                run_dir=run_dir,
                checkpoint=5000,
                num_samples=8,
                gen_device="cpu",
                fair_score_metric=metric,
                fair_score_eo_min_mass=1e-6,
                fair_score_guidance_normalize=normalized,
                fair_sensitive_attr="y",
                fair_sensitive_value=None,
                fair_edge_sensitive_mode="either",
                largest_cc=False,
            )
            generated = build_generate_args(grid_args, eta=eta, k=0.3, seed=2)
            with mock.patch.object(evaluate.torch.cuda, "is_available", return_value=False), \
                    mock.patch.object(evaluate.torch, "manual_seed"), \
                    mock.patch.object(evaluate, "get_data", side_effect=StopBeforeLoadingData) as get_data:
                with self.assertRaises(StopBeforeLoadingData):
                    evaluate.run_evaluate(generated)
            return get_data.call_args.args[0]

    def test_uncontrolled_disables_saved_guidance_for_both_metrics(self):
        for metric in ("eo", "sp"):
            with self.subTest(metric=metric):
                resolved = self.resolve_saved_args(
                    metric=metric, eta=0.0, normalized=True, saved_apply=True,
                )
                self.assertFalse(resolved.fair_score_apply_sample)
                self.assertEqual(resolved.fair_score_eta, 0.0)
                self.assertEqual(resolved.fair_score_metric, metric)

    def test_eo_grid_enables_guidance_despite_saved_false_or_none(self):
        for saved_apply in (False, None):
            for normalized in (False, True):
                with self.subTest(saved_apply=saved_apply, normalized=normalized):
                    resolved = self.resolve_saved_args(
                        metric="eo", eta=0.005, normalized=normalized,
                        saved_apply=saved_apply,
                    )
                    self.assertTrue(resolved.fair_score_sp)
                    self.assertTrue(resolved.fair_score_apply_sample)
                    self.assertEqual(resolved.fair_score_guidance_normalize, normalized)
                    self.assertEqual(resolved.fair_score_metric, "eo")
                    self.assertEqual(resolved.fair_score_eta, 0.005)
                    self.assertEqual(resolved.fair_score_k, 0.3)

    def test_sp_grid_explicitly_overrides_saved_normalization(self):
        resolved = self.resolve_saved_args(
            metric="sp", eta=0.01, normalized=False, saved_apply=False,
        )
        self.assertTrue(resolved.fair_score_sp)
        self.assertTrue(resolved.fair_score_apply_sample)
        self.assertFalse(resolved.fair_score_guidance_normalize)
        self.assertEqual(resolved.fair_score_metric, "sp")


if __name__ == "__main__":
    unittest.main()
