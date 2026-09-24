#!/usr/bin/env python3
"""Reanalyze one historical Appendix artifact, without generation or model fitting.

This is explicitly historical_legacy_pre_only. Its learned k, tracking loss,
T=256 and pilot input-dropout evaluator are preserved and never represented as
the current fixed-k diagnostic protocol. Missing final q cannot be recovered.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from direct_gap_analysis import analyze_snapshot, write_analysis
from direct_gap_runtime import file_hash, graph_hash, object_hash, tensor_hash

ROOT = Path(__file__).resolve().parent
PROTOCOL = "historical_legacy_pre_only_pilot_input_dropout_v1"


def verify_asset(record):
    path = Path(record["path"]).expanduser().resolve()
    if file_hash(path) != record["sha256"]:
        raise ValueError(f"Historical asset hash mismatch: {path}")
    if "bytes" in record and path.stat().st_size != record["bytes"]:
        raise ValueError(f"Historical asset size mismatch: {path}")
    return path


def _pair_codes(value, num_nodes, name):
    pairs = torch.as_tensor(value).detach().cpu()
    if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2,E] with int64 IDs")
    if ((pairs < 0) | (pairs >= num_nodes)).any() or (pairs[0] >= pairs[1]).any():
        raise ValueError(f"{name} must be in-range, canonical i<j pairs")
    codes = (pairs[0] * num_nodes + pairs[1]).numpy()
    if len(np.unique(codes)) != len(codes):
        raise ValueError(f"{name} contains duplicate pairs")
    return pairs, codes


def validate_saved_evaluation(artifact, num_nodes, *, score_atol=2e-6):
    """Validate saved split/labels and decode *only P* from the selected embedding."""
    positive_pairs, positives = _pair_codes(artifact["generated_positive_pairs"], num_nodes, "generated positives")
    pairs, codes = {}, {}
    for name in ("train", "val", "test"):
        pairs[name], codes[name] = _pair_codes(artifact[f"{name}_pairs"], num_nodes, f"{name} pairs")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(codes[left], codes[right]).size:
            raise ValueError(f"Saved {left}/{right} splits overlap")
    membership = {name: np.isin(value, positives) for name, value in codes.items()}
    if not membership["train"].all():
        raise ValueError("Saved training pairs include nonedges of the generated graph")
    positive_partition = np.concatenate([codes[name][membership[name]] for name in ("train", "val", "test")])
    if not np.array_equal(np.sort(positive_partition), np.sort(positives)):
        raise ValueError("Saved positive splits do not partition the completed generated graph")
    labels = torch.as_tensor(artifact["test_labels"]).detach().cpu().reshape(-1)
    if len(labels) != len(codes["test"]) or not torch.isin(labels, torch.tensor([0, 1])).all():
        raise ValueError("Saved test labels are not aligned binary labels")
    if not np.array_equal(labels.numpy().astype(bool), membership["test"]):
        raise ValueError("Saved test labels do not describe the generated graph")
    meta = artifact["gcn_meta"]
    counts = {"train_num_pos": int(membership["train"].sum()),
              "val_num_pos": int(membership["val"].sum()),
              "test_num_pos": int(membership["test"].sum()),
              "test_num_neg": int((~membership["test"]).sum())}
    for key, count in counts.items():
        if meta.get(key) != count:
            raise ValueError(f"Saved evaluator split count mismatch: {key}")
    if counts["train_num_pos"] != int(.8 * len(positives)) or counts["val_num_pos"] != int(.1 * len(positives)):
        raise ValueError("Historical generated-positive split proportions mismatch")
    for name in ("val", "test"):
        if int(membership[name].sum()) != int((~membership[name]).sum()):
            raise ValueError(f"Historical {name} negative-sampling count mismatch")
    embedding = torch.as_tensor(artifact["embedding"]).detach().cpu()
    scores = torch.as_tensor(artifact["test_scores"]).detach().cpu().reshape(-1)
    if embedding.ndim != 2 or embedding.shape[0] != num_nodes or not torch.isfinite(embedding).all():
        raise ValueError("Invalid saved terminal embedding")
    if len(scores) != len(codes["test"]) or not torch.isfinite(scores).all():
        raise ValueError("Invalid saved test scores")
    with torch.no_grad():
        decoded = torch.sigmoid((embedding[pairs["test"][0]] * embedding[pairs["test"][1]]).sum(-1))
    if not torch.allclose(scores, decoded.to(scores.dtype), atol=score_atol, rtol=1e-6):
        raise ValueError("Saved test scores disagree with the selected evaluator embedding / pair IDs")
    return {"split_counts": counts, "test_embedding_score_max_abs_error": float((scores - decoded).abs().max()),
            "test_embedding_score_atol": score_atol, "decoded_pair_count": len(scores),
            "full_cache_gnn_decode": False,
            "split_hash": object_hash({name: tensor_hash(pairs[name]) for name in pairs}),
            "embedding_hash": tensor_hash(embedding), "test_score_hash": tensor_hash(scores),
            "generated_positive_pairs_hash": tensor_hash(positive_pairs)}


def legacy_controller_settings(checkpoint, args, *, target, backbone_T):
    """Read the old learned schedules as they were, without constructing a model."""
    state = checkpoint["controller"]
    if checkpoint.get("policy") != "prespecified_pilot" or checkpoint.get("checkpoint_selection") != "final_epoch":
        raise ValueError("Historical importer requires the recorded final-epoch pilot checkpoint")
    if args.dataset != "cora" or args.fair_score_metric != target or state.get("fair_score_metric") != target:
        raise ValueError("Historical controller target/dataset mismatch")
    if state.get("fair_score_k_mode") != "per_step_sigmoid" or state.get("fair_score_eta_mode") != "per_step_multiplier_softplus":
        raise ValueError("Unsupported historical schedule parameterization")
    if state.get("num_timesteps") != backbone_T or args.diffusion_steps != backbone_T:
        raise ValueError("Historical backbone/controller T mismatch")
    checked = ("seed", "diffusion_steps", "fair_score_learn_k", "fair_score_learn_eta",
               "fair_score_k_tracking_loss_weight", "fair_score_guidance_normalize",
               "fair_score_eta", "fair_score_eta_scale", "fair_score_metric", "controller_epochs")
    for key in checked:
        if checkpoint["args"].get(key) != getattr(args, key):
            raise ValueError(f"Historical checkpoint/args mismatch: {key}")
    if checkpoint["epoch"] != args.controller_epochs - 1:
        raise ValueError("Historical checkpoint is not the final calibration epoch")
    if state["fair_score_eta_base"] != args.fair_score_eta or state["fair_score_eta_scale"] != args.fair_score_eta_scale:
        raise ValueError("Historical eta base/scale mismatch")
    k_raw, eta_raw = (torch.as_tensor(state[key]).detach().cpu() for key in ("fair_score_k_raw", "fair_score_eta_raw"))
    if any(value.shape != (backbone_T,) or not torch.isfinite(value).all() for value in (k_raw, eta_raw)):
        raise ValueError("Invalid historical controller raw schedules")
    k = torch.sigmoid(k_raw)
    eta = args.fair_score_eta * F.softplus(eta_raw) / F.softplus(torch.zeros((), dtype=eta_raw.dtype))
    return {"T": backbone_T, "k_schedule": k.tolist(), "eta_schedule": eta.tolist(),
            "schedule_index": "reverse_t_ascending; t=0 final", "k_mode": state["fair_score_k_mode"],
            "eta_mode": state["fair_score_eta_mode"], "eta_scale": args.fair_score_eta_scale,
            "normalization": args.fair_score_guidance_normalize,
            "learned_k": args.fair_score_learn_k, "learned_eta": args.fair_score_learn_eta,
            "tracking_loss_weight": args.fair_score_k_tracking_loss_weight,
            "calibration_seed": args.seed, "calibration_epochs": args.controller_epochs,
            "checkpoint_epoch": checkpoint["epoch"], "checkpoint_rule": "final_epoch",
            "compatible_with_current_fixed_k_method": False}


def legacy_snapshots(artifact, node_groups, num_nodes, identity):
    support = artifact["observer_support"]
    pairs, codes = _pair_codes(support["pair_ids"], num_nodes, "historical cache")
    if len(codes) != num_nodes * (num_nodes - 1) // 2:
        raise ValueError("Historical artifact does not save the complete pair cache")
    groups = torch.as_tensor(node_groups).reshape(-1)
    if not torch.equal(torch.as_tensor(support["same_mask"]), groups[pairs[0]] == groups[pairs[1]]):
        raise ValueError("Historical same_mask disagrees with reference node metadata/order")
    snapshots = []
    for index, saved in enumerate(artifact["observations"]):
        if saved["target"] != artifact["target"] or saved["total_steps"] != identity["T"]:
            raise ValueError("Historical observation target/T mismatch")
        step = saved["step_number"]
        if saved["loop_index"] != step - 1 or saved["diffusion_t"] != identity["T"] - step:
            raise ValueError("Historical observation reverse-step metadata mismatch")
        if saved["progress"] != step / identity["T"]:
            raise ValueError("Historical observation progress mismatch")
        q = torch.as_tensor(saved["q"]).detach().cpu().clone()
        if q.shape != (len(codes),):
            raise ValueError("Historical q is not a full-cache event snapshot")
        if "phase" in saved and saved["phase"] != "pre":
            raise ValueError("Historical importer refuses to relabel a different snapshot phase")
        snapshot = {**saved, "q": q, "pair_ids": pairs, "same_mask": support["same_mask"],
                    "phase": "pre", "score_name": "q_bar", "reverse_t": saved["diffusion_t"],
                    "snapshot_id": f"{identity['graph_id']}:legacy_pre:{index}",
                    "event_id": f"saved_legacy_observation:{index}", "chunk_index": -1,
                    "chunk_metadata_status": "not_recorded_in_historical_artifact",
                    "snapshot_semantics": "one_saved_cache_event; never_combined_across_events",
                    "visited_mask_status": "not_recorded; all_cache_values_retained",
                    "provenance": {**identity, "snapshot_phase": "pre"}}
        if artifact["target"] == "sp":
            if saved.get("w") is not None:
                raise ValueError("Unexpected w in the audited historical SP artifact")
            snapshot.pop("w", None)
        elif saved.get("w") is None:
            raise ValueError("Historical EO artifact does not contain its actual condition cache")
        else:
            snapshot["w"] = torch.as_tensor(saved["w"]).detach().cpu().clone()
        snapshots.append(snapshot)
    if len(snapshots) != 3 or [s["requested_progress"] for s in snapshots] != [.25, .5, .9]:
        raise ValueError("Expected the three historical Appendix pre-shift events")
    return snapshots


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def import_one(source, target, root_seed, output, *, cpu_threads=1):
    if cpu_threads not in (1, 2):
        raise ValueError("cpu_threads must be 1 or 2")
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path, spec_path = source / "manifest.json", source / "run_spec.json"
    original = json.loads(manifest_path.read_text())
    spec = json.loads(spec_path.read_text())
    if original.get("status") != "complete" or original.get("settings") != spec:
        raise ValueError("Historical manifest/run_spec is incomplete or inconsistent")
    if original.get("policy") != "prespecified_pilot" or spec.get("dataset") != "cora":
        raise ValueError("Only the audited historical Cora pilot is supported")
    target_index = {"sp": 1, "eo": 2}[target]
    if root_seed not in spec["root_seeds"][target_index * 10:(target_index + 1) * 10]:
        raise ValueError("root_seed is not in the recorded target's planned graph roots")
    assets = original["assets"]
    records = {name: assets[name] for name in ("backbone_checkpoint", "backbone_args", "graph")}
    records.update(controller_checkpoint=assets[target]["checkpoint"], controller_args=assets[target]["args_file"])
    paths = {name: verify_asset(record) for name, record in records.items()}
    # Bind each recorded asset to the run spec, preventing a valid but unrelated
    # checkpoint from being substituted into a fabricated manifest envelope.
    spec_paths = {name: spec[name] for name in ("backbone_checkpoint", "backbone_args", "graph")}
    spec_paths.update(controller_checkpoint=spec["controllers"][target]["checkpoint"],
                      controller_args=spec["controllers"][target]["args"])
    for name, value in spec_paths.items():
        if (ROOT / value).resolve() != paths[name]:
            raise ValueError(f"Historical run_spec/asset path mismatch: {name}")
    artifact_path = source / "artifacts" / f"{target}_{root_seed}.pt"
    artifact_digest = file_hash(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("target") != target or artifact.get("root_seed") != root_seed:
        raise ValueError("Historical artifact target/root identity mismatch")
    with paths["backbone_args"].open("rb") as stream:
        backbone_args = pickle.load(stream)
    with paths["controller_args"].open("rb") as stream:
        controller_args = pickle.load(stream)
    checkpoint = torch.load(paths["controller_checkpoint"], map_location="cpu", weights_only=False)
    actual = legacy_controller_settings(checkpoint, controller_args, target=target,
                                        backbone_T=backbone_args.diffusion_steps)
    if actual["calibration_seed"] != spec["pilot"]["train_seeds"][target]:
        raise ValueError("Historical calibration seed mismatch")
    if artifact["gcn_meta"]["split_seed"] != root_seed + 1000000 or artifact["gcn_meta"]["fit_seed"] != root_seed + 2000000:
        raise ValueError("Historical split/evaluator seed mapping mismatch")
    gcn_meta = artifact["gcn_meta"]
    if (gcn_meta["gcn_policy"] != "prespecified_pilot" or gcn_meta["gcn_config"] != spec["gcn"]["config"] or
            gcn_meta["gcn_dropout_application"] != "input_features_training_only"):
        raise ValueError("Historical selected-evaluator protocol mismatch")
    from run_proxy_minimal import read_reference
    reference, _unused_sampler, graph_meta = read_reference(paths["graph"], backbone_args)
    if graph_meta != original["backbone_compatibility"]["graph"]:
        raise ValueError("Reference preprocessing/node order/features/groups differ from historical manifest")
    validation = validate_saved_evaluation(artifact, reference.num_nodes)
    generated = reference.clone()
    positive_pairs = artifact["generated_positive_pairs"].detach().cpu().clone()
    generated.edge_index = torch.cat((positive_pairs, positive_pairs.flip(0)), dim=1)
    generated.edge_attr = None
    order = getattr(reference, "orig_id", None)
    if order is None:
        order = torch.arange(reference.num_nodes)
    identity = {"dataset": "cora", "configuration_id": f"historical_pilot_{target}_final_epoch",
                "graph_id": f"historical:{target}:{root_seed}", "graph_hash": graph_hash(generated),
                "backbone_hash": records["backbone_checkpoint"]["sha256"],
                "controller_hash": records["controller_checkpoint"]["sha256"],
                "node_order_hash": tensor_hash(order), "pair_mapping": "unordered_i_lt_j", "batch_id": 0,
                "controller_seed": actual["calibration_seed"], "graph_seed": root_seed,
                "split_seed": gcn_meta["split_seed"], "protocol": PROTOCOL,
                "generation_mode": "historical_online_generated_graph_reused_offline",
                "historical_legacy_pre_only": True, "T": actual["T"], "variant": "legacy_learned_k_T",
                "normalization": actual["normalization"], "eo_min_mass": controller_args.fair_score_eo_min_mass}
    snapshots = legacy_snapshots(artifact, reference.y, reference.num_nodes, identity)
    evaluation = {"pair_ids": artifact["test_pairs"].clone(), "scores": artifact["test_scores"].clone(),
                  "labels": artifact["test_labels"].clone(), "node_groups": reference.y.clone(),
                  "num_nodes": reference.num_nodes, "generated_positive_pairs": positive_pairs,
                  "node_batch": torch.zeros(reference.num_nodes, dtype=torch.long),
                  "feature_source": "generated_graph.x", "group_source": "generated_graph.y",
                  "feature_hash": tensor_hash(reference.x), "group_hash": tensor_hash(reference.y),
                  "provenance": {**identity, "evaluator_seed": gcn_meta["fit_seed"]},
                  "split_hash": validation["split_hash"], "selected_embedding_hash": validation["embedding_hash"],
                  "evaluator_checkpoint_status": "weights_not_saved; selected_terminal_embedding_saved",
                  "embedding": artifact["embedding"].clone(), "meta": gcn_meta,
                  "train_pairs": artifact["train_pairs"].clone(), "val_pairs": artifact["val_pairs"].clone()}
    trajectory = {"data": generated, "snapshots": snapshots, "provenance": identity, "actual": actual,
                  "q_final_available": False, "unavailable_reason": "not_saved_in_historical_artifact",
                  "historical_source_artifact": {"path": str(artifact_path), "sha256": artifact_digest}}
    rows = []
    for snapshot in snapshots:
        rows.extend(analyze_snapshot(snapshot, evaluation, min_mass=identity["eo_min_mass"]))
    source_code = {}
    for name, record in original.get("sources", {}).items():
        current_path = Path(record["path"])
        current_hash = file_hash(current_path) if current_path.is_file() else None
        source_code[name] = {"recorded": record, "current_sha256": current_hash,
                             "recorded_source_bytes_still_available": current_hash == record["sha256"]}
    source_digests = {str(manifest_path): file_hash(manifest_path), str(spec_path): file_hash(spec_path),
                      str(artifact_path): artifact_digest, **{str(paths[name]): record["sha256"] for name, record in records.items()}}
    output.mkdir(parents=True, exist_ok=False)
    torch.save(trajectory, output / "trajectory.pt")
    torch.save(evaluation, output / "evaluation.pt")
    analysis = write_analysis(rows, output)
    for path, digest in source_digests.items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Historical source changed during reanalysis: {path}")
    resolved = {"schema_version": 1, "mode": "historical_legacy_pre_only", "status": "complete",
                "created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source), "target": target,
                "root_seed": root_seed, "cpu_threads": cpu_threads, "source_assets": records,
                "source_digests": source_digests, "source_assets_unchanged": True,
                "source_hash_binding_limit": "artifact was not individually hashed by old manifest; original co-located artifact identity, splits, scores, support and recorded assets were cross-checked",
                "historical_generation_code_revision": original.get("code_revision", "unavailable_in_legacy_manifest"),
                "historical_source_code": source_code, "actual": actual, "identity": identity,
                "reference_metadata": graph_meta, "evaluator": gcn_meta, "validation": validation,
                "snapshot_phases": ["pre"], "observations": len(snapshots), "generated_graphs_reused": 1,
                "selected_evaluators_reused": 1, "new_graph_generation_count": 0, "new_gnn_fit_count": 0,
                "q_final_available": False, "current_fixed_k_smoke_completed": False,
                "source_snapshot_status": "existing_immutable_full_cache_pre_shift_events",
                "import_code": {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                "importer_sha256": file_hash(__file__), "analyzer_sha256": file_hash(ROOT / "direct_gap_analysis.py")},
                "correlation_status": "undefined; one graph per phase/progress/protocol",
                "analysis_outputs": {key: analysis[key] for key in ("gap_records", "gap_summary", "plot_paths")}}
    _write_json(output / "resolved_manifest.json", resolved)
    _write_json(output / "offline_manifest.json", {"schema_version": 1, "sign_threshold": 1e-6,
                "historical_legacy_pre_only": True,
                "artifacts": [{"trajectory": {"path": "trajectory.pt", "sha256": file_hash(output / "trajectory.pt")},
                               "evaluation": {"path": "evaluation.pt", "sha256": file_hash(output / "evaluation.pt")}}]})
    lines = ["# Historical Cora direct-gap reanalysis", "",
             "Status: complete historical_legacy_pre_only; current fixed-k smoke remains unexecuted.", "",
             f"Reused one generated {target.upper()} graph (root {root_seed}) and its one selected terminal evaluator.",
             f"Actual T={actual['T']}; learned per-step k; tracking weight={actual['tracking_loss_weight']}; "
             "original final-epoch controller and pilot input-dropout evaluator retained.",
             "No generation, controller calibration, backbone training, GNN fitting, or full-cache GNN decoding was performed.",
             "Three original full-cache pre-shift events were read separately; q_final was not saved and is unavailable.",
             "Historical chunk IDs and visit masks were not recorded. No cache values from separate events were combined.",
             "The source artifact had no per-artifact hash in the old manifest. Co-located artifact identity and "
             "recorded checkpoint/data hashes, node metadata, generated splits/labels and selected embedding scores were cross-checked.",
             "One graph verifies offline alignment/arithmetic only; each progress has one graph and correlation is undefined.", "",
             "| progress | a = gap(Q,U) | b = gap(Q,P) | c = gap(g,P) | a-b | b-c | a-c |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["metric"] == "sp":
            lines.append("| " + " | ".join(f"{row[key]:.9g}" for key in (
                "progress", "a", "b", "c", "support_difference", "score_difference", "total_difference")) + " |")
    lines.extend(["", "Values are score units; CSV *_pp columns alone multiply by 100.",
                  "Positive correlation under observed conditions would be empirical alignment, not a causal downstream-descent guarantee.", ""])
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "results/proxy_minimal"))
    parser.add_argument("--target", choices=("sp", "eo"), default="sp")
    parser.add_argument("--root-seed", type=int, default=930011)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    result = import_one(args.source, args.target, args.root_seed, args.output, cpu_threads=args.cpu_threads)
    print(json.dumps({"status": result["status"], "mode": result["mode"], "output": str(Path(args.output).resolve()),
                      "q_final_available": False, "new_gnn_fit_count": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
