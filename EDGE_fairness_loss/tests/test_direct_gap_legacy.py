"""Toy-only validation of legacy import; never executes a historical experiment."""
import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from import_legacy_direct_gap import (legacy_controller_settings, legacy_snapshots,
                                      validate_saved_evaluation, verify_asset)
from direct_gap_runtime import file_hash

torch.set_num_threads(1)


class LegacyImportTests(unittest.TestCase):
    def artifact(self):
        pairs = torch.triu_indices(6, 6, offset=1)
        embedding = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 30
        test_pairs = pairs[:, [9, 11]]
        scores = torch.sigmoid((embedding[test_pairs[0]] * embedding[test_pairs[1]]).sum(-1))
        return {"generated_positive_pairs": pairs[:, :10], "train_pairs": pairs[:, :8],
                "val_pairs": pairs[:, [8, 10]], "test_pairs": test_pairs,
                "test_labels": torch.tensor([1, 0]), "test_scores": scores,
                "embedding": embedding, "gcn_meta": {"train_num_pos": 8, "val_num_pos": 1,
                                                       "test_num_pos": 1, "test_num_neg": 1}}

    def test_saved_generated_split_and_embedding_validate_only_test_pairs(self):
        result = validate_saved_evaluation(self.artifact(), 6)
        self.assertEqual(result["decoded_pair_count"], 2)
        self.assertFalse(result["full_cache_gnn_decode"])
        self.assertEqual(result["test_embedding_score_max_abs_error"], 0)

    def test_misaligned_scores_labels_and_splits_refused(self):
        artifact = self.artifact()
        artifact["test_scores"] = artifact["test_scores"].flip(0)
        with self.assertRaisesRegex(ValueError, "selected evaluator embedding"):
            validate_saved_evaluation(artifact, 6)
        artifact = self.artifact()
        artifact["test_labels"] = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "generated graph"):
            validate_saved_evaluation(artifact, 6)
        artifact = self.artifact()
        artifact["val_pairs"][:, 0] = artifact["train_pairs"][:, 0]
        with self.assertRaisesRegex(ValueError, "splits overlap"):
            validate_saved_evaluation(artifact, 6)
        artifact = self.artifact()
        artifact["train_pairs"] = artifact["train_pairs"][:, :-1]
        with self.assertRaisesRegex(ValueError, "partition"):
            validate_saved_evaluation(artifact, 6)

    def test_recorded_hash_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            path.write_text("old asset")
            record = {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}
            self.assertEqual(verify_asset(record), path)
            path.write_text("changed asset")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_asset(record)

    def test_original_learned_k_tracking_and_eta_schedule_preserved(self):
        args = argparse.Namespace(dataset="cora", fair_score_metric="sp", seed=91, diffusion_steps=3,
                                  fair_score_learn_k=True, fair_score_learn_eta=True,
                                  fair_score_k_tracking_loss_weight=.01, fair_score_guidance_normalize=True,
                                  fair_score_eta=.005, fair_score_eta_scale=1.0, controller_epochs=100)
        checkpoint = {"policy": "prespecified_pilot", "checkpoint_selection": "final_epoch", "epoch": 99,
                      "args": vars(args).copy(), "controller": {
                          "fair_score_metric": "sp", "fair_score_k_mode": "per_step_sigmoid",
                          "fair_score_eta_mode": "per_step_multiplier_softplus", "num_timesteps": 3,
                          "fair_score_eta_base": .005, "fair_score_eta_scale": 1.0,
                          "fair_score_k_raw": torch.tensor([0., .1, .2]),
                          "fair_score_eta_raw": torch.tensor([0., -.1, -.2])}}
        actual = legacy_controller_settings(checkpoint, args, target="sp", backbone_T=3)
        self.assertTrue(actual["learned_k"])
        self.assertEqual(actual["tracking_loss_weight"], .01)
        self.assertFalse(actual["compatible_with_current_fixed_k_method"])
        self.assertEqual(actual["k_schedule"][0], .5)
        self.assertAlmostEqual(actual["eta_schedule"][0], .005)
        self.assertNotEqual(actual["k_schedule"][0], actual["k_schedule"][-1])
        checkpoint["checkpoint_selection"] = "best_validation"
        with self.assertRaisesRegex(ValueError, "final-epoch"):
            legacy_controller_settings(checkpoint, args, target="sp", backbone_T=3)

    def test_saved_events_preserve_pre_phase_no_fabricated_final(self):
        pairs = torch.triu_indices(6, 6, offset=1)
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        original = {"target": "sp", "observer_support": {
            "pair_ids": pairs, "same_mask": groups[pairs[0]] == groups[pairs[1]]}, "observations": []}
        for requested, step in ((.25, 64), (.5, 128), (.9, 231)):
            original["observations"].append({"target": "sp", "total_steps": 256,
                "loop_index": step - 1, "step_number": step, "diffusion_t": 256 - step,
                "progress": step / 256, "requested_progress": requested, "q": torch.full((15,), .2)})
        snapshots = legacy_snapshots(original, groups, 6, {"T": 256, "graph_id": "toy"})
        self.assertEqual(len(snapshots), 3)
        self.assertTrue(all(s["phase"] == "pre" and s["score_name"] == "q_bar" for s in snapshots))
        self.assertTrue(all(s["chunk_index"] == -1 for s in snapshots))
        self.assertTrue(all("w" not in s for s in snapshots))
        original["observations"][0]["q"].zero_()
        self.assertTrue((snapshots[0]["q"] == .2).all())
        wrong = deepcopy(original)
        wrong["observer_support"]["same_mask"][0] = ~wrong["observer_support"]["same_mask"][0]
        with self.assertRaisesRegex(ValueError, "node metadata/order"):
            legacy_snapshots(wrong, groups, 6, {"T": 256, "graph_id": "toy"})
        wrong = deepcopy(original)
        wrong["observations"][0]["phase"] = "post"
        with self.assertRaisesRegex(ValueError, "snapshot phase"):
            legacy_snapshots(wrong, groups, 6, {"T": 256, "graph_id": "toy"})


if __name__ == "__main__":
    unittest.main()
