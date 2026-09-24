"""Small CPU audit cases: corruption/leakage rejected, bad quality accepted."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import proxy_minimal_execution_audit as audit
from proxy_minimal_gcn import EmbeddingDecoder
from proxy_minimal_metrics import analyze_snapshot, terminal_metrics
from proxy_minimal_pilot import PILOT_GCN_CONFIG


def fixture(target="eo", constant=False, single_group=False):
    n, root = 8, 920003
    pairs = torch.triu_indices(n, n, 1)
    groups = torch.zeros(n, dtype=torch.long) if single_group else torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    data = SimpleNamespace(num_nodes=n, y=groups)
    embedding = (torch.ones(n, 2) if constant else torch.arange(16).reshape(n, 2).float() / 20)
    decoder = EmbeddingDecoder(embedding, 5)
    spec = {"gcn": {"config": dict(PILOT_GCN_CONFIG)}, "decode_chunk_size": 5}
    meta = {"gcn_fits": 1, "gcn_config": spec["gcn"]["config"], "gcn_policy": "prespecified_pilot",
            "gcn_optimizer": "Adam", "gcn_dropout_application": "input_features_training_only",
            "gcn_best_epoch": 1, "gcn_epochs_run": 6, "gcn_best_val_auc": 0.2,
            "split_seed": root + 1000000, "fit_seed": root + 2000000,
            "train_num_pos": 16, "val_num_pos": 2, "test_num_pos": 2, "test_num_neg": 2}
    artifact = {"target": target, "root_seed": root, "embedding": embedding,
        "generated_positive_pairs": pairs[:, :20], "train_pairs": pairs[:, :16],
        "val_pairs": pairs[:, [16, 17, 20, 21]], "test_pairs": pairs[:, [18, 19, 22, 23]],
        "test_labels": torch.tensor([1, 1, 0, 0]), "gcn_meta": meta,
        "observer_support": {"pair_ids": pairs, "same_mask": groups[pairs[0]] == groups[pairs[1]]},
        "observations": []}
    artifact["test_scores"] = decoder(artifact["test_pairs"])
    rows = []
    for progress in (() if target == "uncontrolled" else (0.25, 0.5, 0.9)):
        snapshot = {"target": target, "q": torch.linspace(0.9, 0.1, pairs.shape[1]),
                    "requested_progress": progress, "loop_index": __import__("math").ceil(progress * 256) - 1}
        if target == "eo":
            snapshot["w"] = torch.linspace(0.2, 0.7, pairs.shape[1])
        artifact["observations"].append(snapshot)
        rows.append(analyze_snapshot({**artifact["observer_support"], **snapshot}, decode_pairs=decoder,
            generated_positive_pairs=artifact["generated_positive_pairs"],
            test_pairs=artifact["test_pairs"], test_labels=artifact["test_labels"], num_nodes=n, chunk_size=5))
    terminal = terminal_metrics(decoder(artifact["test_pairs"]), artifact["test_labels"],
        artifact["test_pairs"], groups, artifact["generated_positive_pairs"], n,
        generated_positive_scores=decoder(artifact["generated_positive_pairs"]))
    args = {"sp": SimpleNamespace(diffusion_steps=256, fair_score_eo_min_mass=1e-6),
            "eo": SimpleNamespace(diffusion_steps=256, fair_score_eo_min_mass=1e-6)}
    return artifact, {"status": "ok", **terminal}, rows, spec, data, args


class ExecutionAuditTests(unittest.TestCase):
    def test_poor_auc_and_negative_correlation_are_not_quality_gates(self):
        inputs = fixture("sp")
        self.assertLess(inputs[1]["auc"], 0.5)
        self.assertLess(inputs[2][0]["q_g_spearman"], 0)
        report = audit._graph_audit(*inputs)
        self.assertTrue(report["frozen_embedding_reused"])
        self.assertEqual(report["valid_counts"]["q_g_valid_count"], 3)

    def test_constant_correlation_and_missing_groups_preserve_structural_nan(self):
        inputs = fixture(constant=True, single_group=True)
        report = audit._graph_audit(*inputs)
        self.assertEqual(report["valid_counts"]["q_g_valid_count"], 0)
        self.assertEqual(report["valid_counts"]["eo_valid_count"], 0)
        self.assertTrue(report["structural_undefined"])

    def test_split_leakage_rejected(self):
        inputs = fixture()
        inputs[0]["val_pairs"][:, 0] = inputs[0]["train_pairs"][:, 0]
        with self.assertRaisesRegex(ValueError, "leakage"):
            audit._graph_audit(*inputs)

    def test_invalid_group_production_proxy_nan_is_structural(self):
        inputs = fixture(single_group=True)
        artifact, _, rows, _, data, _ = inputs
        decoder = EmbeddingDecoder(artifact["embedding"], 5)
        for i, snapshot in enumerate(artifact["observations"]):
            snapshot.update(production_proxy=torch.tensor([float("nan")]), valid_graph=torch.tensor([False]))
            rows[i] = analyze_snapshot({**artifact["observer_support"], **snapshot}, decode_pairs=decoder,
                generated_positive_pairs=artifact["generated_positive_pairs"],
                test_pairs=artifact["test_pairs"], test_labels=artifact["test_labels"], num_nodes=data.num_nodes, chunk_size=5)
        report = audit._graph_audit(*inputs)
        self.assertEqual(report["valid_counts"]["production_proxy_valid_count"], 0)

    def test_exact_saved_gpu_test_scores_preserve_roundoff_ties(self):
        artifact, _, _, spec, data, _ = fixture(constant=True)
        saved = artifact["test_scores"].clone()
        saved[0] = torch.nextafter(saved[0], torch.ones_like(saved[0]))
        decoder = audit._CheckedDecoder(artifact["embedding"], spec["decode_chunk_size"],
                                        artifact["test_pairs"], saved, data.num_nodes)
        self.assertTrue(torch.equal(decoder(artifact["test_pairs"]), saved))
        saved[0] = 0.0
        with self.assertRaisesRegex(ValueError, "differ from frozen embedding"):
            audit._CheckedDecoder(artifact["embedding"], spec["decode_chunk_size"],
                                   artifact["test_pairs"], saved, data.num_nodes)

    def test_unexpected_nonfinite_and_out_of_range_proxy_rejected(self):
        for key, value in (("q", float("nan")), ("q", 1.2), ("w", float("inf"))):
            inputs = fixture()
            inputs[0]["observations"][0][key][0] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                audit._graph_audit(*inputs)

    def test_corrupted_csv_and_nonfinite_embedding_rejected(self):
        inputs = fixture()
        inputs[2][0]["R"] += 0.1
        with self.assertRaisesRegex(ValueError, "Mismatch.*R"):
            audit._graph_audit(*inputs)
        inputs = fixture()
        inputs[0]["embedding"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite terminal embedding"):
            audit._graph_audit(*inputs)

    def test_all_arms_require_training_only_input_dropout(self):
        for target in ("uncontrolled", "sp", "eo"):
            inputs = fixture(target)
            audit._graph_audit(*inputs)
            inputs[0]["gcn_meta"]["gcn_dropout_application"] = "legacy_hidden_layers_only"
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "dropout"):
                audit._graph_audit(*inputs)

    def test_controller_zero_gradient_fails_without_loading_or_training(self):
        controller_audit = {"backbone_before": "same", "backbone_after": "same",
            "controller_diagnostics": {name: {"grad_missing_epochs": 0, "grad_nonzero_epochs": 0}
                                       for name in audit.runner.CONTROLLER_KEYS}}
        args = SimpleNamespace(dataset="cora")
        spec = {"controllers": {"sp": {"args": "unused", "checkpoint": "unused"}}}
        with mock.patch.object(audit.runner, "load_args", return_value=args), \
             mock.patch.object(audit.runner, "load_checkpoint", return_value={"args": vars(args)}), \
             mock.patch.object(audit.runner, "verify_pilot_final", return_value={"audit": controller_audit}), \
             mock.patch.object(audit.runner, "build_model", side_effect=AssertionError("must stop first")):
            report = audit.validate_controller("sp", spec)
        self.assertEqual(report["status"], "failed")
        self.assertIn("No nonzero", report["errors"][0])
        self.assertEqual(report["new_gcn_fits"], 0)

    def test_wrong_stage_count_stops_before_model_or_cuda(self):
        with tempfile.TemporaryDirectory(prefix="execution_audit_cpu_") as temporary:
            path = Path(temporary)
            spec = {"pilot": {"smoke_seeds": {"uncontrolled": 1, "sp": 2, "eo": 3}}}
            (path / "manifest.json").write_text(json.dumps({"status": "complete", "stage": "smoke",
                "settings": spec, "actual_counts": {"graphs": 4}}))
            with mock.patch.object(audit.runner, "read_reference", side_effect=AssertionError("must stop first")), \
                 mock.patch("torch.cuda.is_available", side_effect=AssertionError("CUDA")):
                report = audit.validate_graph_stage("smoke", spec, path)
            self.assertEqual(report["status"], "failed")
            self.assertIn("Wrong stage count graphs", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
