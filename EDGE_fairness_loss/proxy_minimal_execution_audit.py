"""Read-only CPU gates for the authorized fixed pilot sequence.

These checks never sample a graph, fit a model, access CUDA, or choose settings.
Quality (including negative correlations or worse fairness) is never a gate.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from proxy_minimal_gcn import EmbeddingDecoder
from proxy_minimal_metrics import analyze_snapshot, terminal_metrics
import run_proxy_minimal as runner


def _report(stage):
    return {"stage": stage, "status": "passed", "errors": [], "warnings": [],
            "checks": {}, "valid_counts": {}, "quality_thresholds": None,
            "audit_device": "cpu", "new_graphs": 0, "new_gcn_fits": 0}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _failure(report, label, exc):
    report["status"] = "failed"
    report["errors"].append(f"{label}: {type(exc).__name__}: {exc}")


def validate_controller(target, spec):
    """Check final provenance, exact backbone invariance, gradients and loading."""
    report = _report(f"train-{target}")
    try:
        _require(target in ("sp", "eo"), "Unknown controller target")
        entry = spec["controllers"][target]
        args = runner.load_args(entry["args"])
        checkpoint = runner.load_checkpoint(entry["checkpoint"])
        verified = runner.verify_pilot_final(entry, args, checkpoint, target)
        audit = verified["audit"]
        _require(checkpoint.get("args") == vars(args), "Embedded controller args disagree")
        _require(audit["backbone_before"] == audit["backbone_after"],
                 "Before/after backbone tensor hashes differ")
        for name in sorted(runner.CONTROLLER_KEYS):
            item = audit["controller_diagnostics"][name]
            _require(item.get("grad_missing_epochs") == 0, f"Missing {name} gradients")
            _require(item.get("grad_nonzero_epochs", 0) > 0, f"No nonzero {name} gradient")
            _require(item.get("param_changed") is True, f"Unchanged controller parameter {name}")
            for delta in ("param_delta_l1", "param_delta_max"):
                _require(math.isfinite(item[delta]) and item[delta] > 0,
                         f"Invalid {name} {delta}")
            value = checkpoint["controller"][name]
            _require(bool(torch.isfinite(value).all()), f"Nonfinite final {name}")
            from proxy_minimal_pilot import tensor_fingerprint
            _require(tensor_fingerprint(value) == item["final"],
                     f"Controller tensor differs from audit: {name}")
        backbone_args = runner.load_args(spec["backbone_args"])
        _require(audit.get("device") == spec["device"], "Training used a different CUDA device")
        _require(audit.get("cuda_visible_devices") in (None, ""), "Training remapped CUDA devices")
        _require(audit.get("gpu_telemetry", {}).get("physical_device") == int(spec["device"].split(":")[1]),
                 "Training telemetry physical device differs from plan")
        _require(args.diffusion_steps == backbone_args.diffusion_steps,
                 "Controller diffusion steps differ from backbone")
        _require(args.fair_score_metric == target and
                 checkpoint["controller"]["fair_score_metric"] == target,
                 "Controller target differs")
        _require(runner.file_record(args.controller_pretrained_ckpt)["sha256"] ==
                 runner.file_record(spec["backbone_checkpoint"])["sha256"],
                 "Controller did not start from original backbone")
        data, sampler, _ = runner.read_reference(spec["graph"], backbone_args)
        backbone = runner.model_state(runner.load_checkpoint(spec["backbone_checkpoint"]))
        model = runner.build_model(args, sampler, backbone, checkpoint, device="cpu")
        _require(not model.training and all(not p.requires_grad for p in model.parameters()),
                 "Loaded inference model must be eval and frozen")
        report["checks"].update(final_checkpoint=runner.file_record(entry["checkpoint"]),
                                training_audit=verified["audit_file"],
                                backbone_unchanged=True, controller_diagnostics=audit["controller_diagnostics"],
                                cpu_final_checkpoint_compatible=True, frozen_eval=True,
                                num_nodes=int(data.num_nodes), test_used=False)
        del model, data, sampler, backbone, checkpoint
    except Exception as exc:
        _failure(report, "controller final audit", exc)
    return report


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _probabilities(value, label):
    value = _array(value)
    _require(np.isfinite(value).all(), f"Unexpected nonfinite {label}")
    _require(((value >= 0) & (value <= 1)).all(), f"{label} outside [0, 1]")


def _codes(pairs, n, label):
    pairs = _array(pairs)
    _require(pairs.ndim == 2 and pairs.shape[0] == 2, f"Invalid {label} shape")
    _require(np.issubdtype(pairs.dtype, np.integer), f"Noninteger {label}")
    _require(((pairs >= 0) & (pairs < n)).all(), f"Out-of-range {label}")
    _require((pairs[0] < pairs[1]).all(), f"{label} must contain canonical undirected pairs")
    codes = pairs[0] * n + pairs[1]
    _require(len(np.unique(codes)) == len(codes), f"Duplicate {label}")
    return codes


def _compare_row(actual, expected, label):
    """NaNs match only recomputed structural NaNs after finite inputs are checked."""
    for key, value in expected.items():
        _require(key in actual, f"Missing {label}.{key}")
        observed = actual[key]
        if isinstance(value, (bool, np.bool_)):
            _require(str(observed).lower() == str(bool(value)).lower(), f"Mismatch {label}.{key}")
        elif isinstance(value, (int, float, np.number)):
            numeric = float(observed)
            _require(not math.isinf(numeric), f"Infinite {label}.{key}")
            if math.isnan(float(value)):
                _require(math.isnan(numeric), f"Expected structural NaN {label}.{key}")
            else:
                _require(math.isclose(numeric, float(value), rel_tol=2e-5, abs_tol=2e-7),
                         f"Mismatch {label}.{key}: saved={numeric}, recomputed={value}")
        else:
            _require(str(observed) == str(value), f"Mismatch {label}.{key}")


def _read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def _verify_split(artifact, n):
    positive = _codes(artifact["generated_positive_pairs"], n, "generated positives")
    splits = {name: _codes(artifact[f"{name}_pairs"], n, f"{name} pairs")
              for name in ("train", "val", "test")}
    for i, name in enumerate(splits):
        for other in list(splits)[i + 1:]:
            _require(not len(np.intersect1d(splits[name], splits[other])),
                     f"{name}/{other} pair leakage")
    labels = {name: np.isin(codes, positive) for name, codes in splits.items()}
    count = len(positive)
    expected = {"train": int(count * 0.8), "val": int(count * 0.1)}
    expected["test"] = count - sum(expected.values())
    for name, target in expected.items():
        _require(int(labels[name].sum()) == target, f"Wrong {name} positive split count")
        _require(len(splits[name]) == target * (1 if name == "train" else 2),
                 f"Wrong {name} pair count")
    union = np.concatenate([splits[name][labels[name]] for name in splits])
    _require(np.array_equal(np.sort(union), np.sort(positive)), "Positive split union differs from graph")
    _require(np.array_equal(_array(artifact["test_labels"]), labels["test"].astype(int)),
             "Test labels do not describe generated test positives")
    meta = artifact["gcn_meta"]
    for name, target in expected.items():
        _require(meta[f"{name}_num_pos"] == target, f"Wrong {name} meta count")
    _require(meta["test_num_neg"] == expected["test"], "Wrong test negative meta count")
    return expected


class _CheckedDecoder:
    def __init__(self, embedding, chunk_size, test_pairs, test_scores, num_nodes):
        self.decoder = EmbeddingDecoder(embedding, chunk_size)
        self.num_nodes = num_nodes
        self.test_codes = _codes(test_pairs, num_nodes, "saved test probabilities support")
        order = np.argsort(self.test_codes)
        self.test_codes = self.test_codes[order]
        self.test_scores = torch.as_tensor(test_scores).detach().cpu()[order]
        _probabilities(self.test_scores, "saved exact GPU test probabilities")
        decoded = self.decoder(test_pairs)
        _require(torch.allclose(decoded, torch.as_tensor(test_scores).cpu(), rtol=2e-5, atol=2e-6),
                 "Saved GPU test scores differ from frozen embedding CPU decode")

    def __call__(self, pairs):
        probabilities = self.decoder(pairs)
        _probabilities(probabilities, "frozen terminal GCN probabilities")
        # Preserve the actual GPU probabilities on test pairs. Spearman ranks
        # can otherwise change when CPU/GPU roundoff creates or removes ties.
        array = _array(pairs)
        codes = array[0] * self.num_nodes + array[1]
        indices = np.searchsorted(self.test_codes, codes)
        bounded = np.minimum(indices, len(self.test_codes) - 1)
        selected = (indices < len(self.test_codes)) & (self.test_codes[bounded] == codes)
        probabilities[selected] = self.test_scores[bounded[selected]]
        return probabilities


def _graph_audit(artifact, terminal_row, alignment_rows, spec, data, controller_args):
    target, root = artifact["target"], int(artifact["root_seed"])
    n = int(data.num_nodes)
    _require(terminal_row["status"] == "ok", "Terminal GCN did not complete normally")
    split_counts = _verify_split(artifact, n)
    embedding, meta = artifact["embedding"], artifact["gcn_meta"]
    _require(embedding.ndim == 2 and embedding.shape[0] == n, "Invalid embedding shape")
    _require(bool(torch.isfinite(embedding).all()), "Unexpected nonfinite terminal embedding")
    _require(meta["gcn_fits"] == 1 and meta["gcn_config"] == spec["gcn"]["config"],
             "GCN fit count/config differs from fixed plan")
    _require(meta["gcn_policy"] == "prespecified_pilot" and meta["gcn_optimizer"] == "Adam",
             "Wrong GCN policy/optimizer")
    _require(meta["gcn_dropout_application"] == "input_features_training_only",
             "Pilot dropout not applied uniformly in training only")
    _require(1 <= meta["gcn_best_epoch"] <= meta["gcn_epochs_run"] <= spec["gcn"]["config"]["max_epochs"],
             "GCN epoch counts inconsistent")
    _probabilities([meta["gcn_best_val_auc"]], "best validation AUC")
    _require(meta["split_seed"] == root + 1000000 and meta["fit_seed"] == root + 2000000,
             "GCN seeds differ from plan")
    decoder = _CheckedDecoder(embedding, spec["decode_chunk_size"], artifact["test_pairs"],
                              artifact["test_scores"], n)
    terminal = terminal_metrics(
        decoder(artifact["test_pairs"]), artifact["test_labels"], artifact["test_pairs"],
        data.y, artifact["generated_positive_pairs"], n,
        generated_positive_scores=decoder(artifact["generated_positive_pairs"]))
    _compare_row(terminal_row, terminal, f"{target}/{root}/terminal")
    expected_observations = 0 if target == "uncontrolled" else 3
    _require(len(artifact["observations"]) == len(alignment_rows) == expected_observations,
             "Wrong observation/analysis count")
    valid = {key: int(value) for key, value in terminal.items() if key.endswith("valid_count")}
    structural = []
    for observation in artifact["observations"]:
        support = artifact["observer_support"]
        snapshot = {**support, **observation}
        pairs = snapshot["pair_ids"]
        codes = _codes(pairs, n, "observer full-pair support")
        _require(len(codes) == n * (n - 1) // 2, "Observer does not cover every pair")
        same = data.y[pairs[0]] == data.y[pairs[1]]
        _require(np.array_equal(_array(same), _array(snapshot["same_mask"])), "Observer group mask differs")
        _probabilities(snapshot["q"], "guided running q")
        if target == "eo":
            _probabilities(snapshot["w"], "detached unshifted w")
        if "production_proxy" in snapshot:
            raw_proxy = _array(snapshot["production_proxy"])
            valid_proxy = _array(snapshot.get("valid_graph", [True])).astype(bool)
            _require(not np.isinf(raw_proxy).any() and (np.isfinite(raw_proxy) | ~valid_proxy).all(),
                     "Unexpected nonfinite valid production proxy")
        progress = float(snapshot["requested_progress"])
        rows = [row for row in alignment_rows if float(row["requested_progress"]) == progress]
        _require(progress in runner.PROGRESS and len(rows) == 1, "Duplicate or wrong observer progress")
        steps = int(controller_args[target].diffusion_steps)
        _require(snapshot["loop_index"] == math.ceil(progress * steps) - 1,
                 "Observer timing differs from prescribed pre-guidance step")
        min_mass = controller_args[target].fair_score_eo_min_mass
        measured = analyze_snapshot(
            snapshot, decode_pairs=decoder, generated_positive_pairs=artifact["generated_positive_pairs"],
            test_pairs=artifact["test_pairs"], test_labels=artifact["test_labels"], num_nodes=n,
            chunk_size=spec["decode_chunk_size"], min_mass=min_mass)
        _compare_row(rows[0], measured, f"{target}/{root}/{progress}")
        if measured["identity_valid_count"]:
            _require(measured["identity_holds"], "R/C/S telescoping identity failed")
        for key, value in measured.items():
            if key.endswith("valid_count"):
                valid[key] = valid.get(key, 0) + int(value)
        for key in ("d", "R", "C", "S", "terminal_gap", "q_g_spearman"):
            if math.isnan(measured[key]):
                structural.append({"progress": progress, "metric": key,
                    "reason": "constant/fewer-than-two correlation pairs" if key == "q_g_spearman"
                    else "missing group or weight mass at/below min_mass"})
    return {"target": target, "root_seed": root, "split_positive_counts": split_counts,
            "frozen_embedding_reused": True, "score_dropout": False,
            "exact_saved_test_scores_for_correlations": True,
            "valid_counts": valid, "structural_undefined": structural}


def validate_graph_stage(stage, spec, output_dir):
    """Validate existing smoke/E1 files with CPU chunk decoding; no extra fits."""
    report = _report(stage)
    try:
        _require(stage in ("smoke", "e1"), "Unknown graph stage")
        output = Path(output_dir)
        manifest = json.loads((output / "manifest.json").read_text())
        _require(manifest["status"] == "complete" and manifest["stage"] == stage,
                 "Stage is not complete")
        _require(manifest["settings"] == spec, "Stage settings differ from approved spec")
        roots = ({target: [seed] for target, seed in spec["pilot"]["smoke_seeds"].items()}
                 if stage == "smoke" else
                 {target: spec["root_seeds"][i * 10:(i + 1) * 10]
                  for i, target in enumerate(("uncontrolled", "sp", "eo"))})
        expected_keys = {(target, int(root)) for target, seeds in roots.items() for root in seeds}
        expected_counts = {"graphs": 3 if stage == "smoke" else 30,
                           "gcn_fits": 3 if stage == "smoke" else 30,
                           "gcn_fit_attempts": 3 if stage == "smoke" else 30,
                           "observations": 6 if stage == "smoke" else 60}
        for name, count in expected_counts.items():
            _require(manifest["actual_counts"][name] == count, f"Wrong stage count {name}")
        _require(manifest["cost"]["gpu_telemetry"]["physical_device"] == int(spec["device"].split(":")[1]),
                 "Stage telemetry physical GPU differs from plan")
        for asset in ("backbone_checkpoint", "backbone_args", "graph"):
            _require(manifest["assets"][asset]["sha256"] == runner.file_record(spec[asset])["sha256"],
                     f"Original asset changed: {asset}")
        for target in ("sp", "eo"):
            _require(manifest["assets"][target]["checkpoint"]["sha256"] ==
                     runner.file_record(spec["controllers"][target]["checkpoint"])["sha256"],
                     f"Final controller changed: {target}")
        terminal = _read_csv(output / "terminal_metrics.csv")
        alignment = _read_csv(output / "proxy_alignment.csv")
        terminal_keys = [(row["target"], int(row["root_seed"])) for row in terminal]
        _require(len(terminal_keys) == len(expected_keys) and set(terminal_keys) == expected_keys,
                 "Terminal roots/arms differ from prescribed stage or include smoke in E1")
        _require(len(alignment) == expected_counts["observations"], "Wrong alignment CSV count")
        expected_alignment = {(target, root, progress) for target, root in expected_keys
                              if target != "uncontrolled" for progress in runner.PROGRESS}
        alignment_keys = [(row["target"], int(row["root_seed"]), float(row["requested_progress"]))
                          for row in alignment]
        _require(set(alignment_keys) == expected_alignment, "Alignment roots/progress differ")
        artifact_paths = list((output / "artifacts").glob("*.pt"))
        _require({p.name for p in artifact_paths} == {f"{t}_{r}.pt" for t, r in expected_keys},
                 "Artifact count or names differ")
        reference_args = runner.load_args(spec["backbone_args"])
        data, _sampler, _record = runner.read_reference(spec["graph"], reference_args)
        controller_args = {target: runner.load_args(spec["controllers"][target]["args"])
                           for target in ("sp", "eo")}
        report["checks"].update(counts=expected_counts, artifact_hashes={}, graphs=[],
                                generated_test_positives_for_eo=True,
                                dropout_disabled_for_all_saved_embedding_scores=True)
        for target, root in sorted(expected_keys):
            label = f"{target}/{root}"
            try:
                path = output / "artifacts" / f"{target}_{root}.pt"
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                _require(artifact["target"] == target and int(artifact["root_seed"]) == root,
                         "Artifact identity differs from filename")
                terminal_row = next(row for row in terminal
                                    if row["target"] == target and int(row["root_seed"]) == root)
                alignment_rows = [row for row in alignment
                                  if row["target"] == target and int(row["root_seed"]) == root]
                graph_report = _graph_audit(artifact, terminal_row, alignment_rows, spec,
                                            data, controller_args)
                report["checks"]["graphs"].append(graph_report)
                report["checks"]["artifact_hashes"][label] = runner.file_record(path)
                for key, value in graph_report["valid_counts"].items():
                    report["valid_counts"][key] = report["valid_counts"].get(key, 0) + value
                if graph_report["structural_undefined"]:
                    report["warnings"].append(f"{label}: structural NaNs preserved; see graph checks")
                del artifact
            except Exception as exc:
                _failure(report, label, exc)
        report["checks"]["quality_used_for_gate"] = False
    except Exception as exc:
        _failure(report, "stage artifact audit", exc)
    return report
