#!/usr/bin/env python3
"""Cora / EDGE / FairShift-T E1 only. Default: CPU preflight, zero experiments."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

from proxy_minimal_gcn import InvalidTerminalFit, fit_terminal_gcn, seed_all, validate_config
from proxy_minimal_pilot import start_gpu_telemetry, update_gpu_telemetry


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/proxy_minimal"
CONTROLLER_KEYS = {"fair_score_k_raw", "fair_score_eta_raw"}
PROGRESS = (0.25, 0.5, 0.9)
STRICT_POLICY = "validation_selected"
PILOT_POLICY = "prespecified_pilot"


def resolve_path(value):
    if not value:
        raise ValueError("Missing path")
    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if any("fairwire" in part.lower() for part in path.parts):
        raise ValueError("FairWire assets are outside E1 scope")
    if not (ROOT in path.parents or (ROOT.parent / "EDGE_fairness") in path.parents):
        raise ValueError(f"Asset outside the two authorized repositories: {path}")
    return path


def file_record(value):
    path = resolve_path(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def load_checkpoint(value):
    return torch.load(resolve_path(value), map_location="cpu", weights_only=False)


def load_args(value):
    with resolve_path(value).open("rb") as stream:
        args = pickle.load(stream)
    if args.dataset != "cora":
        raise ValueError("Only Cora is supported")
    return args


def model_state(checkpoint):
    state = checkpoint.get("model", checkpoint)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    return {k: v for k, v in state.items() if k not in CONTROLLER_KEYS}


def read_reference(path, args):
    from datasets.data_utils import preprocess, EmpiricalEmptyGraphGenerator
    with resolve_path(path).open("rb") as stream:
        graph = pickle.load(stream)
    data = preprocess(graph, degree=args.degree)
    if data.x is None or data.x.shape != (graph.number_of_nodes(), args.num_node_feat):
        raise ValueError("Checkpoint feature dimension disagrees with graph")
    if max(d for _, d in graph.degree()) != args.max_degree:
        raise ValueError("Checkpoint max_degree disagrees with graph")
    if args.empty_graph_sampler != "empirical" or args.augmented_features:
        raise ValueError("E1 requires the existing Cora empirical sampler without feature augmentation")
    sampler = EmpiricalEmptyGraphGenerator([data], degree=args.degree,
                                           augment_features=args.augmented_features)
    nodes = list(graph.nodes())
    return data, sampler, {"num_nodes": data.num_nodes, "num_edges": graph.number_of_edges(),
                          "features_shape": list(data.x.shape),
                          "node_order_sha256": hashlib.sha256(repr(nodes).encode()).hexdigest(),
                          "features_sha256": hashlib.sha256(data.x.numpy().tobytes()).hexdigest(),
                          "groups_sha256": hashlib.sha256(data.y.numpy().tobytes()).hexdigest(),
                          "full_pair_count": data.num_nodes * (data.num_nodes - 1) // 2}


def build_model(args, sampler, backbone_state, controller=None, *, device="cpu"):
    from model import get_model
    args = argparse.Namespace(**vars(args))
    args.device = device
    if controller is None:
        args.fair_score_controller_train = False
        args.fair_score_eta = 0.0
    else:
        args.fair_score_controller_train = True
    model = get_model(args, sampler)
    incompatible = model.load_state_dict(backbone_state, strict=False)
    allowed_missing = CONTROLLER_KEYS if controller is not None else set()
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise ValueError(f"Backbone key incompatibility: {incompatible}")
    if controller is not None:
        model.load_fair_controller_state_dict(controller, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


def selection_evidence(selection, expected_basis, roots):
    if not isinstance(selection, dict) or selection.get("basis") != expected_basis:
        raise ValueError(f"Missing documented selection basis {expected_basis}")
    if not selection.get("description"):
        raise ValueError("Selection record needs a description of the independent validation procedure")
    record = file_record(selection.get("evidence_file"))
    selection_roots = selection.get("root_seeds")
    if (not isinstance(selection_roots, list) or not selection_roots
            or any(type(seed) is not int for seed in selection_roots)):
        raise ValueError("Selection trajectory root seeds must be documented")
    if set(selection_roots) & set(roots):
        raise ValueError("E1 roots overlap selection trajectories")
    return {**selection, "evidence": record}


def operating_point_evidence(spec, selection, expected_basis, roots):
    """Keep validation provenance mandatory unless the explicit pilot is used."""
    if spec.get("policy", STRICT_POLICY) != PILOT_POLICY:
        return selection_evidence(selection, expected_basis, roots)
    if (not isinstance(selection, dict) or selection.get("basis") != PILOT_POLICY
            or selection.get("uses_test_for_selection") is not False):
        raise ValueError("Pilot requires prespecified_pilot and uses_test_for_selection=false")
    if expected_basis == "validation" and selection.get("checkpoint_rule") != "final_epoch":
        raise ValueError("Pilot controllers must use the final epoch, without operating-point selection")
    return dict(selection)


def validate_pilot_plan(spec):
    from proxy_minimal_pilot import (PILOT_CONTROLLER_CONFIG, PILOT_GCN_CONFIG,
                                     PILOT_SMOKE_SEEDS, PILOT_TRAIN_SEEDS)
    pilot = spec.get("pilot", {})
    expected = {"controller_config": PILOT_CONTROLLER_CONFIG, "train_seeds": PILOT_TRAIN_SEEDS,
                "smoke_seeds": PILOT_SMOKE_SEEDS, "uses_test_for_selection": False}
    if pilot != expected:
        raise ValueError("Pilot training settings/seeds must equal the prespecified policy")
    if spec.get("gcn", {}).get("config") != PILOT_GCN_CONFIG:
        raise ValueError("Pilot requires its single prespecified GCN config")
    if spec.get("root_seeds") != list(range(930001, 930031)):
        raise ValueError("Pilot preserves E1 roots 930001 through 930030")
    for target in ("sp", "eo"):
        run = OUTPUT / "controllers" / target / f"cora_prespecified_pilot_{target}"
        for key, expected_path in (("checkpoint", run / "check/controller_final.pt"),
                                   ("args", run / "args.pickle"),
                                   ("training_audit", run / "pilot_training_audit.json")):
            if resolve_path(spec["controllers"][target][key]) != expected_path:
                raise ValueError(f"Pilot {target} must reference its one new run's {key}")
    groups = [set(PILOT_TRAIN_SEEDS.values()), set(PILOT_SMOKE_SEEDS.values()),
              set(spec["root_seeds"]), {s + 1000000 for s in spec["root_seeds"]},
              {s + 2000000 for s in spec["root_seeds"]},
              {s + 1000000 for s in PILOT_SMOKE_SEEDS.values()},
              {s + 2000000 for s in PILOT_SMOKE_SEEDS.values()}]
    for i, first in enumerate(groups):
        if any(first & second for second in groups[i + 1:]):
            raise ValueError("Training, smoke and E1 random streams overlap")


def verify_pilot_final(entry, args, checkpoint, target):
    from proxy_minimal_pilot import PILOT_CONTROLLER_CONFIG, PILOT_TRAIN_SEEDS
    if resolve_path(entry["checkpoint"]).name != "controller_final.pt":
        raise ValueError("Pilot explicitly loads controller_final.pt, never best/last")
    if checkpoint.get("epoch") != 99 or not getattr(args, "prespecified_pilot", False):
        raise ValueError("Pilot checkpoint must be the final epoch of its one 100-epoch run")
    if args.seed != PILOT_TRAIN_SEEDS[target]:
        raise ValueError("Pilot controller has a different training seed")
    for key, value in PILOT_CONTROLLER_CONFIG.items():
        if getattr(args, key) != value:
            raise ValueError(f"Pilot controller setting changed: {key}")
    audit_path = resolve_path(entry["training_audit"])
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "complete" or audit.get("policy") != PILOT_POLICY:
        raise ValueError("Pilot training audit is not complete")
    if audit.get("target") != target or audit.get("seed") != args.seed:
        raise ValueError("Pilot audit target/seed disagrees with the final checkpoint")
    if audit.get("configuration") != PILOT_CONTROLLER_CONFIG:
        raise ValueError("Pilot training audit has different prespecified settings")
    if audit.get("completed_epochs") != 100 or audit.get("final_epoch") != 99:
        raise ValueError("Pilot training did not complete exactly 100 epochs")
    for flag in ("test_used", "real_reference_evaluator_constructed", "automatic_best_reload", "hyperparameter_grid"):
        if audit.get(flag) is not False:
            raise ValueError(f"Pilot audit must verify {flag}=false")
    expected_counts = {"optimizer_steps": 100, "replay_sample_calls": 10, "replay_graphs": 20,
                       "auto_export_graphs": 0, "evaluation_graphs": 0, "gcn_fits": 0, "retries": 0}
    if audit.get("counts") != expected_counts:
        raise ValueError("Pilot audit exceeds or disagrees with the prespecified training budget")
    for asset, path in (("backbone_checkpoint", args.controller_pretrained_ckpt),
                        ("backbone_args", args.pilot_backbone_args), ("graph", args.pilot_graph)):
        if audit.get("assets", {}).get(asset, {}).get("sha256") != file_record(path)["sha256"]:
            raise ValueError(f"Pilot training asset hash changed: {asset}")
    if audit.get("backbone_unchanged") is not True:
        raise ValueError("Pilot training must verify exact backbone tensor invariance")
    if set(audit.get("trainable_names", [])) != CONTROLLER_KEYS:
        raise ValueError("Pilot trained parameters other than the two controller schedules")
    if audit.get("final_checkpoint", {}).get("sha256") != file_record(entry["checkpoint"])["sha256"]:
        raise ValueError("Final checkpoint hash differs from the training audit")
    for key in CONTROLLER_KEYS:
        diagnostics = audit.get("controller_diagnostics", {}).get(key, {})
        if not diagnostics.get("grad_finite") or not diagnostics.get("param_finite"):
            raise ValueError(f"Pilot controller had nonfinite/unverified gradients: {key}")
    return {"audit_file": file_record(entry["training_audit"]), "audit": audit}


def pilot_cost_ledger(spec):
    """Separate the preparation budget from the immutable E1 graph budget."""
    training = {}
    for target in ("sp", "eo"):
        entry = spec["controllers"][target]
        path = resolve_path(entry["training_audit"])
        actual = json.loads(path.read_text()) if path.exists() else None
        training[target] = {"seed": spec["pilot"]["train_seeds"][target],
            "planned": {"runs": 1, "optimizer_steps": 100, "replay_sample_calls": 10,
                        "replay_graphs": 20, "auto_export_graphs": 0, "evaluation_graphs": 0,
                        "gcn_fits": 0},
            "actual": actual, "final_checkpoint": file_record(entry["checkpoint"])
                       if resolve_path(entry["checkpoint"]).is_file() else {"path": str(resolve_path(entry["checkpoint"])), "sha256": "TBD"}}
    smoke_path = OUTPUT / "smoke/manifest.json"
    smoke_actual = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    return {"policy": PILOT_POLICY, "controller_training": training,
            "historical_incomplete_execution": json.loads((OUTPUT / "historical_costs.json").read_text())
                if (OUTPUT / "historical_costs.json").is_file() else None,
            "smoke": {"planned": {"graphs": 3, "gcn_fits": 3, "observations": 6},
                      "root_seeds": spec["pilot"]["smoke_seeds"],
                      "actual": {"status": smoke_actual["status"], "counts": smoke_actual["actual_counts"],
                                 "cost": smoke_actual["cost"]} if smoke_actual else None},
            "E1": {"graphs": 30, "gcn_fits": 30, "observations": 60},
            "automatic_grid": False, "automatic_retraining": False, "retries": 0,
            "unmeasured_time_and_peak_vram": "TBD"}


def preflight(spec):
    policy = spec.get("policy", STRICT_POLICY)
    pilot = policy == PILOT_POLICY
    manifest = {
        "status": "dry_run", "scope": "Cora-EDGE-FairShift-T-E1",
        "policy": policy, "blockers": [], "assets": {}, "settings": spec,
        "planned_counts": {"uncontrolled": 10, "sp": 10, "eo": 10,
                           "graphs": 30, "gcn_fits": 30, "observations": 60},
        "actual_counts": {"graphs": 0, "gcn_fits": 0, "observations": 0,
                          "gcn_fit_attempts": 0, "gpu_smoke": 0, "retries": 0},
        "actual_stages": ["cpu_preflight"],
        "evaluator_protocol": {
            "function": "proxy_minimal_gcn.fit_terminal_gcn", "hyperparameter_grid": False,
            "fits_per_graph": 1, "reuse_embedding_for_all_observations": True,
            "positive_split": "generated upper triangle: floor(.8 M), floor(.1 M), remainder",
            "message_passing": "symmetric train adjacency with self loops; sparse",
            "validation_test_negatives": "disjoint true generated nonedges, one per positive",
            "training_negatives": "uniform ordered pairs (legacy GAE rule, possible self/positive collisions)",
            "features": "unchanged reference x in reference node order; no label concatenation",
            "SP": "signed same-minus-different mean probability on generated test pairs",
            "EO": "signed same-minus-different mean probability on generated test positives",
            "uncertainty_unit": "one independent generated graph/root; no pooled edges/timesteps",
            "invalid": "NaN and valid counts; no seed replacement",
            "device": "cuda:4", "decode_chunk_size": spec.get("decode_chunk_size"),
        },
        "cost": {"gpu_hours": "TBD before execution",
                 "money": "TBD", "extra_gpu_smoke_or_retries_authorized": 0},
    }
    blockers = manifest["blockers"]
    if policy not in (STRICT_POLICY, PILOT_POLICY):
        blockers.append("Unknown operating point policy")
    if pilot:
        try:
            validate_pilot_plan(spec)
            manifest["pilot_preparation"] = {"policy_validated": True,
                "validation_selected_operating_point": False, "uses_test_for_selection": False,
                "controller_rule": "one new run per target, final epoch only"}
        except Exception as exc:
            blockers.append(f"Pilot policy: {exc}")
    roots = spec.get("root_seeds", [])
    if len(roots) != 30 or len(set(roots)) != 30 or any(type(s) is not int or not 0 <= s < 2**31 for s in roots):
        blockers.append("Exactly 30 distinct nonnegative integer root seeds are required")
    if spec.get("dataset") != "cora" or spec.get("device") != "cuda:4":
        blockers.append("This E1 runner requires dataset=cora and physical device=cuda:4")
    if set(spec.get("controllers", {})) != {"sp", "eo"}:
        blockers.append("controllers must contain exactly sp and eo; baseline is always uncontrolled")
    if type(spec.get("decode_chunk_size")) is not int or spec["decode_chunk_size"] < 1:
        blockers.append("decode_chunk_size must be a positive integer")
    backbone_args = backbone_state = sampler = None
    for name in ("backbone_checkpoint", "backbone_args", "graph"):
        try:
            manifest["assets"][name] = file_record(spec.get(name))
        except Exception as exc:
            blockers.append(f"{name}: {exc}")
    try:
        backbone_args = load_args(spec["backbone_args"])
        backbone_state = model_state(load_checkpoint(spec["backbone_checkpoint"]))
        _data, sampler, graph_meta = read_reference(spec["graph"], backbone_args)
        baseline = build_model(backbone_args, sampler, backbone_state)
        manifest["backbone_compatibility"] = {"strict_noncontroller_keys": True,
            "state_tensor_count": len(backbone_state), "args": vars(backbone_args),
            "frozen": all(not p.requires_grad for p in baseline.parameters()), "graph": graph_meta}
        steps = int(baseline.num_timesteps)
        manifest["planned_observation_stages"] = [
            {"requested_progress": p, "loop_index": math.ceil(p * steps) - 1,
             "step_number": math.ceil(p * steps), "diffusion_t": steps - math.ceil(p * steps)}
            for p in PROGRESS]
        manifest["cost"].update({"reverse_steps_total": 30 * steps,
            "formula_seconds": "10*t_uncontrolled + 10*t_sp + 10*t_eo + 30*t_gcn + 60*t_analysis",
            "observer_tensor_disk_estimate_bytes": graph_meta["full_pair_count"] * (20 * 17 + 90 * 4),
            "disk_estimate_note": "shared int64 pair IDs/bool masks per guided graph plus float32 q/w; excludes embeddings/logs"})
        del baseline
    except Exception as exc:
        blockers.append(f"CPU backbone construction/compatibility: {type(exc).__name__}: {exc}")
    for target in ("sp", "eo"):
        try:
            entry = spec.get("controllers", {}).get(target)
            if not entry:
                raise ValueError(f"No existing validation-selected T-{target.upper()} checkpoint configured")
            evidence = operating_point_evidence(spec, entry.get("selection"), "validation", roots)
            if pilot:
                # These are planned new outputs, never resolved to discovered legacy controllers.
                expected = {key: str(resolve_path(entry[key])) for key in ("checkpoint", "args", "training_audit")}
                missing = [key for key, path in expected.items() if not Path(path).is_file()]
                if missing:
                    manifest.setdefault("pending_controller_artifacts", {})[target] = expected
                    raise FileNotFoundError(f"Pilot controller training pending: {', '.join(missing)}")
            ckpt = load_checkpoint(entry["checkpoint"])
            args = load_args(entry["args"])
            state = ckpt["controller"]
            if ckpt.get("args") != vars(args):
                raise ValueError("Controller checkpoint's embedded args differ from its args.pickle")
            if (state.get("fair_score_k_mode") != "per_step_sigmoid"
                    or state.get("fair_score_eta_mode") != "per_step_multiplier_softplus"):
                raise ValueError("Controller schedule parameterization is incompatible")
            if args.seed in roots:
                raise ValueError("E1 roots overlap controller training seed")
            if state.get("fair_score_metric", "sp") != target or args.fair_score_metric != target:
                raise ValueError("Controller target disagrees with args/checkpoint")
            if state["num_timesteps"] != backbone_args.diffusion_steps or args.diffusion_steps != backbone_args.diffusion_steps:
                raise ValueError("Controller/backbone diffusion step mismatch")
            if state["fair_score_eta_base"] != args.fair_score_eta:
                raise ValueError("Controller base eta disagrees with its saved args")
            if state.get("fair_score_eo_min_mass", 1e-6) != args.fair_score_eo_min_mass:
                raise ValueError("Controller EO minimum mass disagrees with its saved args")
            if state.get("fair_score_eta_scale", 1.0) != args.fair_score_eta_scale:
                raise ValueError("Controller eta scale disagrees with its saved args")
            # The old loader only restores schedules: other semantics come from saved args.
            for key in ("degree", "augmented_features", "empty_graph_sampler", "fair_label_attr",
                        "noise_schedule", "parametrization", "final_prob_edge", "predict_s",
                        "active_method", "active_ratio", "active_threshold", "diffusion_stage",
                        "use_node_feat", "has_node_feature", "num_node_classes", "num_edge_classes",
                        "num_node_feat", "max_degree", "diffusion_dim", "num_heads", "norm",
                        "final_prob_node", "augmented_feature_dict", "dp_rate", "edge_dropout",
                        "node_feat_dropout", "node_feat_mask_prob", "degree_t_jitter", "degree_0_jitter",
                        "degree_t_mask_prob", "degree_0_mask_prob", "global_context_dropout"):
                if getattr(args, key) != getattr(backbone_args, key):
                    raise ValueError(f"Shared backbone/data/sampling argument mismatch: {key}")
            linked = file_record(args.controller_pretrained_ckpt)
            if linked["sha256"] != manifest["assets"]["backbone_checkpoint"]["sha256"]:
                raise ValueError("Controller points to a different backbone")
            candidate = build_model(args, sampler, backbone_state, ckpt)
            manifest["assets"][target] = {
                "checkpoint": file_record(entry["checkpoint"]), "args_file": file_record(entry["args"]),
                "args": vars(args), "epoch": ckpt.get("epoch"), "selection": evidence,
                "strict_schedule_shapes": {key: list(state[key].shape) for key in CONTROLLER_KEYS},
                "frozen": all(not p.requires_grad for p in candidate.parameters())}
            if pilot:
                manifest["assets"][target]["training_verification"] = verify_pilot_final(entry, args, ckpt, target)
            del candidate
        except Exception as exc:
            blockers.append(f"T-{target.upper()}: {type(exc).__name__}: {exc}")
    try:
        gcn = spec.get("gcn")
        if not gcn:
            raise ValueError("No independently validation-selected fixed GCN setting configured")
        manifest["fixed_gcn"] = {"config": validate_config(gcn["config"]),
            "selection": operating_point_evidence(spec, gcn.get("selection"), "independent_validation", roots)}
        if pilot:
            manifest["fixed_gcn"]["dropout_placement"] = "input features, p=0.1 in training only"
            manifest["evaluator_protocol"]["pilot_difference_from_existing_evaluator"] = {
                "input_dropout_probability": 0.1,
                "training_applies_identically_to": ["uncontrolled", "sp", "eo"],
                "dropout_disabled_for": ["validation", "test", "proxy_analysis"],
                "existing_evaluator": "intermediate-layer dropout only; no input dropout for its 1-layer GCN",
            }
    except Exception as exc:
        blockers.append(f"GCN: {type(exc).__name__}: {exc}")
    audit_path = OUTPUT / "asset_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        manifest["asset_audit"] = ({"backbone": audit.get("backbone"), "graph": audit.get("graph"),
            "historical_controller_candidates_not_used": True} if pilot else audit)
    if pilot and manifest.get("pilot_preparation", {}).get("policy_validated"):
        manifest["preparation_costs"] = pilot_cost_ledger(spec)
        preparation_path = OUTPUT / "controller_preparation.json"
        if preparation_path.is_file():
            manifest["pilot_preparation"]["cpu_model_probes"] = json.loads(preparation_path.read_text())
        commands_path = OUTPUT / "commands.json"
        if commands_path.is_file():
            manifest["planned_commands"] = json.loads(commands_path.read_text())
        approved = manifest.get("planned_commands", {}).get("gpu_execution_approved", False)
        manifest["cost"]["extra_gpu_smoke_or_retries_authorized"] = {
            "gpu_execution_approved": approved, "separate_smoke_graphs": 3, "retries": 0}
    manifest["sources"] = {path: file_record(path) for path in (
        "README.md", "run_proxy_minimal.py", "proxy_minimal_gcn.py", "proxy_minimal_metrics.py",
        "diffusion/diffusion_binomial_active.py", "diffusion/fairness_surrogate.py",
        "evaluate_generated_graphs.py", "model.py", "datasets/data_utils.py")}
    if pilot:
        for source in ("proxy_minimal_pilot.py", "proxy_minimal_execution_audit.py",
                       "train_controller.py", "results/proxy_minimal/commands.sh"):
            if (ROOT / source).is_file():
                manifest["sources"][source] = file_record(source)
    validation_path = OUTPUT / "validation.json"
    if validation_path.exists():
        manifest["cpu_validation"] = json.loads(validation_path.read_text())
    manifest["environment"] = {"python": sys.executable, "torch": torch.__version__,
                               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                               "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
                               "dry_run_uses_gpu": False}
    manifest["status"] = "blocked" if blockers else "ready_awaiting_gpu_approval"
    if pilot and manifest.get("pending_controller_artifacts") and all("Pilot controller training pending" in reason for reason in blockers):
        manifest["status"] = "ready_for_controller_training_awaiting_gpu_approval"
        manifest["e1_ready"] = False
    if pilot and manifest.get("planned_commands", {}).get("gpu_execution_approved"):
        manifest["status"] = manifest["status"].replace("_awaiting_gpu_approval", "_gpu_approved")
    return manifest


def write_csv(path, rows, defaults):
    keys = list(dict.fromkeys([*defaults, *(key for row in rows for key in row)]))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(manifest, terminal, alignment, output_dir=None):
    from proxy_minimal_metrics import summarize_rows
    output_dir = OUTPUT if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    write_csv(output_dir / "terminal_metrics.csv", terminal, ["target", "root_seed", "status", "auc", "sp", "sp_abs", "eo", "eo_abs"])
    write_csv(output_dir / "proxy_alignment.csv", alignment, ["target", "root_seed", "requested_progress", "loop_index", "d", "R", "C", "S"])
    summary = summarize_rows(terminal, alignment)
    write_csv(output_dir / "terminal_summary.csv", summary["terminal"], ["arm", "graph_count"])
    write_csv(output_dir / "alignment_summary.csv", summary["alignment"], ["target", "requested_progress", "graph_count"])
    (output_dir / "aggregate.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    lines = ["# Cora / EDGE / FairShift-T E1", "", f"Status: {manifest['status']}", "",
        f"Policy: {manifest.get('policy', STRICT_POLICY)}. Stage: {manifest.get('stage', 'e1')}.",
        f"Planned counts: {json.dumps(manifest.get('planned_counts', {}))}",
        f"Actual counts: {json.dumps(manifest['actual_counts'])}", "",
        "This evaluator performs no controller training; pilot preparation costs are separate. No E2 branching, snapshot/resume, F, FairWire, extra datasets or LaTeX changes.", "",
        "## Preflight blockers", "", *[f"- {item}" for item in manifest["blockers"]], "",
        "## Budget", "", f"{json.dumps(manifest['cost'], ensure_ascii=False)}", "",
        "## Graph/root aggregates", "", "```json", json.dumps(summary, indent=2, default=str), "```", "",
        "Invalid metrics remain NaN with valid counts. Three observations share one terminal GCN fit.",
        "This diagnoses score/group-moment alignment only; E2 and causal guidance effects are untested.", ""]
    (output_dir / "summary.md").write_text("\n".join(lines))
    if manifest.get("policy") == PILOT_POLICY and manifest.get("pilot_preparation", {}).get("policy_validated"):
        ledger = pilot_cost_ledger(manifest["settings"])
        (OUTPUT / "pilot_costs.json").write_text(json.dumps(ledger, indent=2, default=str) + "\n")


def check_smoke_completed(spec):
    path = OUTPUT / "smoke/manifest.json"
    if not path.is_file():
        raise ValueError("Run the separately budgeted smoke stage before pilot E1")
    smoke = json.loads(path.read_text())
    if smoke.get("status") != "complete" or smoke.get("stage") != "smoke":
        raise ValueError("Smoke has not completed; no automatic retries")
    if smoke.get("settings") != spec:
        raise ValueError("The pilot settings changed after smoke")
    if (smoke["actual_counts"]["graphs"] != 3 or smoke["actual_counts"]["observations"] != 6
            or smoke["actual_counts"]["gcn_fit_attempts"] != 3):
        raise ValueError("Smoke did not cover the three prescribed roots and six guided observations")
    for asset in ("backbone_checkpoint", "backbone_args", "graph"):
        if smoke["assets"][asset]["sha256"] != file_record(spec[asset])["sha256"]:
            raise ValueError(f"Original asset changed after smoke: {asset}")
    for target in ("sp", "eo"):
        if smoke["assets"][target]["checkpoint"]["sha256"] != file_record(spec["controllers"][target]["checkpoint"])["sha256"]:
            raise ValueError("Final controller changed after smoke")
    # No AUC/fairness/proxy quality threshold is used to select or rerun the pilot.


def execute(spec, manifest, *, stage="e1"):
    from evaluate_generated_graphs import unique_undirected_edge_index
    from proxy_minimal_metrics import analyze_snapshot, terminal_metrics
    if manifest["blockers"]:
        raise ValueError("Preflight blockers must be resolved before GPU execution")
    pilot = spec.get("policy", STRICT_POLICY) == PILOT_POLICY
    if stage not in ("e1", "smoke") or (stage == "smoke" and not pilot):
        raise ValueError("Only the prespecified pilot has a separate smoke stage")
    if pilot and stage == "e1":
        check_smoke_completed(spec)
    output_dir = OUTPUT / "smoke" if stage == "smoke" else OUTPUT
    manifest["stage"] = stage
    if stage == "smoke":
        manifest["planned_counts"] = {"uncontrolled": 1, "sp": 1, "eo": 1,
                                      "graphs": 3, "gcn_fits": 3, "observations": 6}
        manifest["cost"] = {"gpu_hours": "TBD", "budget_category": "preparation_smoke", "retries": 0}
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ValueError("Unset CUDA_VISIBLE_DEVICES so cuda:4 denotes physical GPU 4")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 5:
        raise ValueError("Physical cuda:4 unavailable; no fallback device")
    device = "cuda:4"
    torch.cuda.set_device(4)
    manifest["cost"]["gpu_telemetry"] = start_gpu_telemetry(4)
    manifest["status"] = "running"
    if stage == "smoke":
        manifest["actual_counts"]["gpu_smoke"] = 1
    run_start = time.perf_counter()
    manifest["actual_stages"].append("gpu_smoke" if stage == "smoke" else "gpu_E1")
    manifest["environment"]["gpu"] = torch.cuda.get_device_name(device)
    backbone_args = load_args(spec["backbone_args"])
    backbone = model_state(load_checkpoint(spec["backbone_checkpoint"]))
    data, sampler, _ = read_reference(spec["graph"], backbone_args)
    terminal, alignment = [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=False)
    write_outputs(manifest, terminal, alignment, output_dir)
    for index, target in enumerate(("uncontrolled", "sp", "eo")):
        entry = None if target == "uncontrolled" else spec["controllers"][target]
        args = load_args(entry["args"]) if entry else backbone_args
        controller = load_checkpoint(entry["checkpoint"]) if entry else None
        model = build_model(args, sampler, backbone, controller, device=device)
        seeds = ([spec["pilot"]["smoke_seeds"][target]] if stage == "smoke"
                 else spec["root_seeds"][index * 10:(index + 1) * 10])
        for root in seeds:
            identity = {"target": target, "root_seed": root}
            observations = []
            seed_all(root, device)
            start = time.perf_counter()
            with torch.no_grad():
                generated = model.sample(1, proxy_observer=observations.append if entry else None,
                                         proxy_observer_progress=PROGRESS).cpu().to_data_list()[0]
            torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - start
            manifest["actual_counts"]["graphs"] += 1
            manifest["actual_counts"]["observations"] += len(observations)
            if entry and len(observations) != 3:
                raise RuntimeError("Expected exactly three actual guided observations")
            for key in ("x", "y", "orig_id"):
                if getattr(data, key, None) is not None and not torch.equal(getattr(generated, key), getattr(data, key)):
                    raise RuntimeError(f"Generated graph changed reference {key}/node order")
            positives = unique_undirected_edge_index(generated.edge_index)
            # Save generated edges and observation data, never copy reference x/backbone/data files.
            support = {key: observations[0][key] for key in ("pair_ids", "same_mask")} if observations else None
            compact = [{key: value for key, value in observation.items()
                        if key not in ("pair_ids", "same_mask", "pair_batch")}
                       for observation in observations]
            artifact = {**identity, "generated_positive_pairs": positives,
                        "observer_support": support, "observations": compact}
            artifact_path = artifact_dir / f"{target}_{root}.pt"
            torch.save(artifact, artifact_path)
            fit_start = time.perf_counter()
            manifest["actual_counts"]["gcn_fit_attempts"] += 1
            try:
                fit_policy = {"policy": PILOT_POLICY} if pilot else {}
                result = fit_terminal_gcn(generated, spec["gcn"]["config"], split_seed=root + 1000000,
                    fit_seed=root + 2000000, device=device, chunk_size=spec["decode_chunk_size"], **fit_policy)
            except InvalidTerminalFit as exc:
                manifest["actual_counts"]["gcn_fits"] += exc.fits
                invalid_metrics = terminal_metrics([], [], torch.empty((2, 0), dtype=torch.long),
                    data.y, positives, data.num_nodes)
                terminal.append({**identity, "status": "invalid", "reason": str(exc),
                    **invalid_metrics,
                    "gcn_fits": exc.fits,
                    "generation_seconds": generation_seconds,
                    "gcn_seconds": time.perf_counter() - fit_start})
                for observation in observations:
                    alignment.append({**identity, "requested_progress": observation["requested_progress"],
                        "loop_index": observation["loop_index"], "status": "invalid_gcn",
                        **{key: 0 for key in ("d_valid_count", "R_valid_count", "C_valid_count",
                                             "S_valid_count", "q_g_valid_count", "identity_valid_count",
                                             "terminal_gap_valid_count", "signed_agreement_valid_count")},
                        **{key: float("nan") for key in ("d", "R", "C", "S", "terminal_gap",
                            "R_abs", "C_abs", "S_abs", "q_g_spearman", "signed_agreement", "identity_residual")}})
                write_outputs(manifest, terminal, alignment, output_dir)
                del observations, artifact, generated, compact, support
                if entry:
                    del observation
                continue
            manifest["actual_counts"]["gcn_fits"] += 1
            scores = terminal_metrics(test_scores=result["test_scores"], test_labels=result["test_labels"],
                test_pairs=result["test_pairs"], node_groups=data.y,
                generated_positive_pairs=positives, num_nodes=data.num_nodes,
                generated_positive_scores=result["decoder"](positives))
            terminal.append({**identity, "status": "ok", **scores, **result["meta"],
                "generation_seconds": generation_seconds, "gcn_seconds": time.perf_counter() - fit_start})
            for observation in observations:
                analysis_start = time.perf_counter()
                row = analyze_snapshot(observation, decode_pairs=result["decoder"],
                    generated_positive_pairs=positives, test_pairs=result["test_pairs"],
                    test_labels=result["test_labels"], num_nodes=data.num_nodes,
                    chunk_size=spec["decode_chunk_size"], min_mass=args.fair_score_eo_min_mass)
                alignment.append({**identity, **row, "analysis_seconds": time.perf_counter() - analysis_start})
            artifact.update({"embedding": result["embedding"].cpu(), "train_pairs": result["train_pairs"],
                             "val_pairs": result["val_pairs"], "test_pairs": result["test_pairs"],
                             "test_labels": result["test_labels"], "test_scores": result["test_scores"].cpu(),
                             "gcn_meta": result["meta"]})
            torch.save(artifact, artifact_path)
            manifest["cost"]["elapsed_seconds"] = time.perf_counter() - run_start
            write_outputs(manifest, terminal, alignment, output_dir)
            print(f"{stage} {manifest['actual_counts']['graphs']}/{manifest.get('planned_counts', {}).get('graphs', 30)}: {target} root={root}", flush=True)
            del result, observations, artifact, generated, compact, support
            if entry:
                del observation
        del model
    manifest["status"] = "complete"
    manifest["cost"]["elapsed_seconds"] = time.perf_counter() - run_start
    manifest["cost"]["gpu_hours"] = manifest["cost"]["elapsed_seconds"] / 3600
    manifest["cost"]["gpu_hours_definition"] = "Stage wall hours on the reserved GPU, including CPU work; not pure kernel time"
    update_gpu_telemetry(manifest["cost"].get("gpu_telemetry"))
    manifest["actual_stages"].append("graph_root_aggregation")
    write_outputs(manifest, terminal, alignment, output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="results/proxy_minimal/run_spec.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="CPU preflight only (default)")
    mode.add_argument("--execute", action="store_true", help="Run exactly 30 graphs, only after user GPU approval")
    mode.add_argument("--smoke", action="store_true", help="Pilot-only: three separate smoke roots/fits; requires GPU approval")
    cli = parser.parse_args(argv)
    os.chdir(ROOT)
    torch.set_num_threads(4)
    output_dir = OUTPUT / "smoke" if cli.smoke else OUTPUT
    if (output_dir / "artifacts").exists():
        parser.error("An E1 run already has artifacts; overwrite/resume/retry are not implemented")
    spec = json.loads(resolve_path(cli.spec).read_text())
    manifest = preflight(spec)
    if cli.smoke:
        manifest["stage"] = "smoke"
    write_outputs(manifest, [], [], output_dir)
    if cli.execute or cli.smoke:
        execution_start = time.perf_counter()
        try:
            execute(spec, manifest, stage="smoke" if cli.smoke else "e1")
        except BaseException as exc:
            # Preserve existing partial rows and account for the failed attempt;
            # never rerun a graph, controller or smoke automatically.
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["cost"]["elapsed_seconds"] = time.perf_counter() - execution_start
            manifest["cost"]["gpu_hours"] = manifest["cost"]["elapsed_seconds"] / 3600
            manifest["cost"]["gpu_hours_definition"] = "Stage wall hours including CPU work; no GPU interval exists if CUDA setup was not reached"
            update_gpu_telemetry(manifest["cost"].get("gpu_telemetry"))
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
            with (output_dir / "summary.md").open("a") as stream:
                stream.write(f"\nExecution failed without retry: {manifest['error']}\n")
            if manifest.get("pilot_preparation", {}).get("policy_validated"):
                (OUTPUT / "pilot_costs.json").write_text(json.dumps(pilot_cost_ledger(spec), indent=2, default=str) + "\n")
            raise
    print(json.dumps({"status": manifest["status"], "blockers": manifest["blockers"],
                      "manifest": str(output_dir / "manifest.json")}, indent=2))
    return 2 if manifest["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
