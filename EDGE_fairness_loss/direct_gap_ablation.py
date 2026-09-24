"""Opt-in matched fixed-k ablation; observe never imports or calls calibration.

Both arms use the existing replay fairness + utility objective with tracking
weight exactly zero. k=1 removes temporal blending while preserving the full
persistent cache. Calibration is separate from subsequent online generation.
"""
from copy import deepcopy
import json
from pathlib import Path
import random
import types

import numpy as np
import torch
import torch.nn.functional as F

from direct_gap_runtime import (build_observed_model, file_hash, object_hash,
                                tensor_hash, validate_controller)


CALIBRATION_FIELDS = (
    "k0", "variant", "seed", "epochs", "lr", "replay_samples", "replay_refresh",
    "fairness_weight", "utility_weight", "eta_init", "clip_value", "clip_norm",
)


def validate_calibration(spec):
    calibration = deepcopy(spec.get("calibration", {}))
    missing = [key for key in CALIBRATION_FIELDS if key not in calibration]
    if missing:
        raise ValueError("Explicit matched calibration settings required: " + ", ".join(missing))
    if calibration["variant"] not in ("T", "tied"):
        raise ValueError("Calibration variant must be T or tied")
    for key in ("seed", "epochs", "replay_samples", "replay_refresh"):
        value = calibration[key]
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid calibration {key}")
    if not 0 <= calibration["seed"] < 2 ** 32 - calibration["epochs"]:
        raise ValueError("Calibration/replay seeds must fit numpy's uint32 seed range")
    for key in ("k0", "lr", "fairness_weight", "utility_weight", "eta_init"):
        value = calibration[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid calibration {key}")
    if not 0 < calibration["k0"] < 1:
        raise ValueError("k0 must be in (0,1); the paired arm uses exact k=1")
    if min(calibration["lr"], calibration["eta_init"]) <= 0:
        raise ValueError("lr and eta_init must be positive")
    if calibration["fairness_weight"] + calibration["utility_weight"] <= 0:
        raise ValueError("At least one existing objective weight must be positive")
    for key in ("clip_value", "clip_norm"):
        value = calibration[key]
        if value is not None and (type(value) not in (int, float) or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be null or finite positive")
    for key in ("tracking_loss_weight", "learned_k", "uses_test_for_selection"):
        if key in calibration and calibration[key] not in (0, False):
            raise ValueError(f"{key} is forbidden in matched fixed-k calibration")
    configurations = spec.get("configurations", [])
    if not configurations:
        raise ValueError("At least one explicit F initialization configuration is required")
    for config in configurations:
        controller = config.get("controller", {})
        if controller.get("kind") != "F":
            raise ValueError("Matched calibration must start from an explicit F configuration, not a selected controller")
        if controller.get("uses_test_for_selection") is not False:
            raise ValueError("Initialization must explicitly exclude test-based selection")
        if controller.get("k") != calibration["k0"] or controller.get("eta") != calibration["eta_init"]:
            raise ValueError("F initialization k/eta must exactly match calibration k0/eta_init")
        if controller.get("metric") != config.get("target") or config.get("target") not in ("sp", "eo"):
            raise ValueError("Configuration and initialization metrics must match")
        if type(controller.get("normalization")) is not bool:
            raise ValueError("Normalization must be explicit and unchanged across arms")
    return calibration


def prepare_plan(spec, code_revision):
    """Resolve both arms without instantiating a model or starting calibration."""
    calibration = validate_calibration(spec)
    result = deepcopy(spec)
    result["calibration"] = calibration
    result["ablation_protocol"] = "matched_fixed_k_eta_calibration_v1"
    result["ablation_code_revision"] = code_revision
    result["ablation_plan"] = []
    for config in spec["configurations"]:
        assets = {}
        for key in ("backbone_checkpoint", "backbone_args", "graph"):
            path = Path(config[key])
            assets[key] = {"path": str(path), "sha256": file_hash(path) if path.is_file() else None}
        for name, k in (("k0", calibration["k0"]), ("k1", 1.0)):
            result["ablation_plan"].append({
                "id": f"{config['id']}__{name}", "source_configuration": config["id"],
                "fixed_k": k, "variant": calibration["variant"], "assets": assets,
                "calibration": deepcopy(calibration), "status": "planned_not_executed",
                "interpretation": "no temporal blending; persistent cache retained" if k == 1 else "fixed temporal blending",
                "tracking_loss_weight": 0.0, "learned_k": False,
                "uses_test_for_selection": False, "checkpoint_rule": "last_epoch_only",
                "calibration_mode": "existing_unguided_stage1_replay",
                "evaluation_mode": "separate_explicit_online_generation",
            })
    return result


def _seed(seed, device):
    from proxy_minimal_gcn import seed_all
    seed_all(seed, device)


def enable_eta_only(model, *, eta_init, variant):
    """Keep exact fixed-k adapter and expose only the existing positive eta form."""
    model.eval().requires_grad_(False)
    if variant == "T":
        native_eta = getattr(model, "fair_score_eta_raw", None)
        if (getattr(model, "fair_score_eta_mode", "per_step") == "per_step"
                and native_eta is not None and tuple(native_eta.shape) == (model.num_timesteps,)):
            parameter = native_eta
            with torch.no_grad():
                parameter.zero_()
            parameter.requires_grad_(True)
            parameter_name = "fair_score_eta_raw"
        else:
            # A backbone's saved shared-eta mode is not the requested T variant.
            # Keep that native mode/tensor frozen and give this explicit adapter
            # its own per-step parameter instead of rewriting the native model.
            parameter = torch.nn.Parameter(torch.zeros(model.num_timesteps, device=model.device))
            model.register_parameter("_direct_gap_per_step_eta_raw", parameter)
            parameter_name = "_direct_gap_per_step_eta_raw"
    elif variant == "tied":
        parameter = torch.nn.Parameter(torch.zeros((), device=model.device))
        model.register_parameter("_direct_gap_tied_eta_raw", parameter)
        parameter_name = "_direct_gap_tied_eta_raw"
    else:
        raise ValueError("variant must be T or tied")

    def eta(self, *args, **kwargs):
        raw = getattr(self, parameter_name)
        # This is the existing controller's eta_init * softplus(raw)/softplus(0).
        values = float(eta_init) * F.softplus(raw) / F.softplus(raw.new_zeros(()))
        if variant == "tied":
            values = values.expand(self.num_timesteps)
        index = kwargs.get("t_graph")
        return values if index is None else values[index.to(values.device).long()]

    model._get_effective_fair_score_eta = types.MethodType(eta, model)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable != [parameter_name]:
        raise RuntimeError(f"Calibration unexpectedly exposes other parameters: {trainable}")
    return parameter_name, parameter


def _frozen_digest(model, parameter_name):
    return object_hash({name: tensor_hash(value) for name, value in model.state_dict().items()
                        if name != parameter_name})


def calibrate_arm(config, calibration, *, fixed_k, device, code_revision,
                  model_builder=None):
    """Bounded eta calibration, used only by the explicit opt-in command."""
    model_builder = model_builder or build_observed_model
    _seed(calibration["seed"], device)
    arm_config = deepcopy(config)
    arm_config["controller"].update({"k": fixed_k, "eta": calibration["eta_init"]})
    model, data, actual = model_builder(arm_config, device=device)
    parameter_name, parameter = enable_eta_only(
        model, eta_init=calibration["eta_init"], variant=calibration["variant"])
    model.fair_score_fair_loss_weight = calibration["fairness_weight"]
    model.fair_score_utility_loss_weight = calibration["utility_weight"]
    model.fair_score_k_tracking_loss_weight = 0.0
    frozen_before = _frozen_digest(model, parameter_name)
    expected_k = torch.full((model.num_timesteps,), float(fixed_k), device=model.device)
    if not torch.equal(model._get_effective_fair_score_k(), expected_k):
        raise RuntimeError("Fixed-k adapter does not preserve the requested exact k")
    optimizer = torch.optim.Adam([parameter], lr=calibration["lr"])
    trace, replay_seeds = [], []
    replay = None
    for epoch in range(calibration["epochs"]):
        if epoch % calibration["replay_refresh"] == 0:
            replay_seed = calibration["seed"] + epoch
            replay_seeds.append(replay_seed)
            _seed(replay_seed, device)
            with torch.no_grad():
                _, replay = model.sample(calibration["replay_samples"], return_controller_replay=True)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = model.compute_fair_controller_loss_from_replay(replay)
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"Nonfinite calibration loss at epoch {epoch}; no checkpoint exported")
        loss.backward()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Invalid eta gradient at epoch {epoch}; no checkpoint exported")
        if calibration["clip_value"] is not None:
            torch.nn.utils.clip_grad_value_([parameter], calibration["clip_value"])
        if calibration["clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_([parameter], calibration["clip_norm"])
        optimizer.step()
        if not torch.equal(model._get_effective_fair_score_k(), expected_k):
            raise RuntimeError("k changed during eta-only calibration")
        trace.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu()), **stats})
    frozen_after = _frozen_digest(model, parameter_name)
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen backbone/controller state changed during eta calibration")
    eta_schedule = model._get_effective_fair_score_eta().detach().cpu().tolist()
    checkpoint = {
        "format": "fairshift_fixed_k_eta_v1", "dataset": config["dataset"],
        "backbone_hash": actual["backbone_hash"], "num_timesteps": model.num_timesteps,
        "fixed_k": float(fixed_k), "eta_schedule": eta_schedule, "variant": calibration["variant"],
        "trainable_parameters": ["eta_t"] if calibration["variant"] == "T" else ["eta"],
        "tracking_loss_weight": 0.0, "learned_k": False, "uses_test_for_selection": False,
        "metric": config["target"], "normalization": model.fair_score_guidance_normalize,
        "calibration_seed": calibration["seed"], "checkpoint_rule": "last_epoch_only",
        "objective": {"name": "existing_compute_fair_controller_loss_from_replay",
                      "fairness_weight": calibration["fairness_weight"],
                      "utility_weight": calibration["utility_weight"], "tracking_loss_weight": 0.0},
        "budget": {key: calibration[key] for key in ("epochs", "lr", "replay_samples", "replay_refresh", "clip_value", "clip_norm")},
        "code_revision": code_revision, "calibration": deepcopy(calibration),
        "replay_seeds": replay_seeds, "calibration_mode": "existing_unguided_stage1_replay",
        "generation_mode": "not_generated_by_calibration",
        "frozen_state_hash_before": frozen_before, "frozen_state_hash_after": frozen_after,
        "backbone_unchanged": True, "k_semantics": "no temporal blending; persistent cache retained" if fixed_k == 1 else "fixed temporal blending",
        "feature_hash": tensor_hash(data.x),
        "group_hash": tensor_hash(getattr(data, model.fair_label_attr).reshape(-1).long()),
        "group_label_attr": model.fair_label_attr,
        "fair_label_attr": model.fair_label_attr,
        "eo_min_mass": float(model.fair_score_eo_min_mass),
        "backbone_args_hash": file_hash(config["backbone_args"]),
        "reference_graph_hash": file_hash(config["graph"]),
        "feature_source": "backbone preprocess node metadata (not edge evaluation targets)",
        "group_source": f"backbone preprocess node metadata.{model.fair_label_attr}",
        "source_asset_hashes": {key: file_hash(config[key]) for key in ("backbone_checkpoint", "backbone_args", "graph")},
        "resolved_initial_model": actual,
    }
    checkpoint = validate_controller(checkpoint, backbone_hash=actual["backbone_hash"],
                                     timesteps=model.num_timesteps, dataset=config["dataset"])
    return checkpoint, trace


def run_ablation(args):
    from run_direct_gap import code_identity, fresh_output, resolve_manifest, write_json

    if args.command not in ("prepare-k-ablation", "calibrate-k-ablation"):
        raise ValueError("Ablation only runs through its explicit command")
    if args.command == "calibrate-k-ablation" and not args.allow_calibration:
        raise ValueError("Calibration requires both calibrate-k-ablation and --allow-calibration")
    spec = resolve_manifest(args.manifest)
    revision = code_identity()
    plan = prepare_plan(spec, revision)
    # A prepared manifest binds the exact assets, rather than accepting replacements.
    if "ablation_plan" in spec:
        old_assets = {arm["id"]: arm["assets"] for arm in spec["ablation_plan"]}
        new_assets = {arm["id"]: arm["assets"] for arm in plan["ablation_plan"]}
        if old_assets != new_assets:
            raise ValueError("Prepared ablation asset hashes changed")
    output = fresh_output(args.output)
    write_json(output / "matched_calibration_manifest.json", plan)
    if args.command == "prepare-k-ablation":
        print(json.dumps({"status": "prepared_not_executed", "output": str(output),
                          "arms": len(plan["ablation_plan"])}))
        return 0
    device = spec.get("device", "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU unavailable; no fallback or calibration performed")
    observe = deepcopy(spec)
    observe.pop("ablation_plan", None)
    observe["configurations"] = []
    by_id = {config["id"]: config for config in spec["configurations"]}
    statuses = []
    # Run every planned arm, retaining failures; never select by a resulting gap.
    for arm in plan["ablation_plan"]:
        directory = output / arm["id"]
        directory.mkdir(exist_ok=False)
        try:
            if any(value["sha256"] is None for value in arm["assets"].values()):
                raise ValueError("Required backbone/graph assets are missing")
            config = by_id[arm["source_configuration"]]
            checkpoint, trace = calibrate_arm(config, plan["calibration"], fixed_k=arm["fixed_k"],
                                              device=device, code_revision=revision)
            path = directory / "controller_last.pt"
            with path.open("xb") as stream:
                torch.save(checkpoint, stream)
            write_json(directory / "calibration_trace.json", trace)
            entry = deepcopy(config)
            entry["id"] = arm["id"]
            suffix = "k1" if arm["fixed_k"] == 1.0 else "k0"
            source_configuration_id = config.get("configuration_id", config["id"])
            entry["configuration_id"] = f"{source_configuration_id}__{suffix}"
            entry["controller"] = {"kind": checkpoint["variant"], "checkpoint": str(path)}
            entry["ablation"] = {"protocol": plan["ablation_protocol"], "fixed_k": arm["fixed_k"],
                                 "source_configuration_id": source_configuration_id,
                                 "matched_calibration": True, "checkpoint_sha256": file_hash(path)}
            observe["configurations"].append(entry)
            statuses.append({"id": arm["id"], "status": "calibrated_not_generated", "checkpoint": str(path)})
        except Exception as exc:
            failure = {"id": arm["id"], "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            statuses.append(failure)
            write_json(directory / "failure.json", failure)
    complete = all(status["status"] == "calibrated_not_generated" for status in statuses)
    # Do not offer a partial matched experiment as a complete observation manifest.
    if complete:
        write_json(output / "observe_manifest.json", observe)
    write_json(output / "calibration_report.json", {
        "status": "complete_not_generated" if complete else "incomplete", "arms": statuses,
        "no_online_generation_performed": True, "no_evaluator_training_performed": True,
        "planned_arms": len(plan["ablation_plan"]), "k1_retains_persistent_cache": True,
    })
    return 0 if complete else 2
