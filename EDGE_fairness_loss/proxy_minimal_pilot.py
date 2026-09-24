"""Prespecified Cora pilot policy and audited, frozen-backbone controller training.

This opt-in path never constructs a reference evaluator, searches settings,
reloads a best checkpoint, exports generated graphs, or trains the backbone.
Only controller parameters are persisted; the sibling backbone stays in place.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch


def start_gpu_telemetry(device=4):
    """Reset allocator peaks before model loading, without sampling or RNG use."""
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "physical_device": device,
        "reservation_started_monotonic_seconds": time.monotonic(),
        "time_definition": "GPU reservation wall interval including CPU work and idle gaps; not pure kernel time",
        "memory_definition": "Exact PyTorch process allocator peaks since reset before model loading; excludes CUDA context and non-PyTorch allocations",
        "status": "started",
    }


def update_gpu_telemetry(telemetry):
    """Keep telemetry errors visible without hiding the original run failure."""
    if not telemetry:
        return
    telemetry["gpu_reservation_wall_seconds"] = (
        time.monotonic() - telemetry["reservation_started_monotonic_seconds"])
    telemetry["gpu_reservation_wall_hours"] = telemetry["gpu_reservation_wall_seconds"] / 3600
    try:
        telemetry["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(telemetry["physical_device"])
        telemetry["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(telemetry["physical_device"])
        telemetry["status"] = "measured"
    except Exception as exc:
        telemetry["status"] = "measurement_error"
        telemetry["error"] = f"{type(exc).__name__}: {exc}"


POLICY = "prespecified_pilot"
ROOT = Path(__file__).resolve().parent
CONTROLLER_KEYS = {"fair_score_k_raw", "fair_score_eta_raw"}
PILOT_CONTROLLER_CONFIG = {
    "controller_epochs": 100,
    "controller_lr": 5e-4,
    "fair_score_k": 0.5,
    "fair_score_eta": 0.005,
    "fair_score_fair_loss_weight": 1.0,
    "fair_score_utility_loss_weight": 0.1,
    "fair_score_k_tracking_loss_weight": 0.01,
    "controller_replay_num_samples": 2,
    "controller_replay_refresh": 10,
    "fair_score_guidance_normalize": True,
}
PILOT_GCN_CONFIG = {
    "num_layers": 1, "hidden_size": 128, "dropout": 0.1,
    "lr": 0.01, "weight_decay": 0.0, "max_epochs": 1000,
    "patience": 5, "batch_size": 16384, "min_delta": 0.0,
    "selection_metric": "validation_auc",
}
PILOT_TRAIN_SEEDS = {"sp": 910001, "eo": 910002}
PILOT_SMOKE_SEEDS = {"uncontrolled": 920001, "sp": 920002, "eo": 920003}


def _path(value, *, output=False):
    if not value:
        raise ValueError("Pilot asset/output path is required")
    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if any("fairwire" in part.lower() for part in path.parts):
        raise ValueError("FairWire paths are outside this pilot")
    allowed = [ROOT] if output else [ROOT, ROOT.parent / "EDGE_fairness"]
    if not any(root == path or root in path.parents for root in allowed):
        raise ValueError(f"Pilot path is outside authorized {'output' if output else 'input'} roots: {path}")
    return path


def file_record(value):
    path = _path(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def validate_physical_gpu4(device):
    """Validate addressing only; safe in CPU preparation with no CUDA calls."""
    if str(device) != "cuda:4":
        raise ValueError("The prespecified pilot requires physical cuda:4")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise ValueError("Unset CUDA_VISIBLE_DEVICES to address physical cuda:4 without remapping")


def prepare_pilot_args(cli_args, supplied_options=None):
    """Preserve saved architecture/data args and overlay this pilot's settings.

    Parser defaults never overwrite the saved backbone configuration. If CLI
    options are provided, reject architecture overrides and changed pilot values.
    """
    validate_physical_gpu4(cli_args.device)
    target = cli_args.fair_score_metric
    if target not in PILOT_TRAIN_SEEDS or cli_args.seed != PILOT_TRAIN_SEEDS[target]:
        raise ValueError(f"Pilot training requires the fixed target seed {PILOT_TRAIN_SEEDS}")
    allowed = set(PILOT_CONTROLLER_CONFIG) | {
        "prespecified_pilot", "pilot_backbone_args", "pilot_graph", "pilot_audit_path",
        "controller_pretrained_ckpt", "controller_root", "fair_score_metric", "seed", "name", "device",
    }
    supplied = set(supplied_options or ())
    unknown = supplied - allowed
    if unknown:
        raise ValueError(f"Pilot CLI cannot override saved architecture/data or add workflows: {sorted(unknown)}")
    for key, value in PILOT_CONTROLLER_CONFIG.items():
        if key in supplied and getattr(cli_args, key) != value:
            raise ValueError(f"Pilot fixes {key}={value!r}")
    paths = {key: _path(getattr(cli_args, key, None)) for key in (
        "pilot_backbone_args", "pilot_graph", "controller_pretrained_ckpt")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = (ROOT.parent / "EDGE_fairness/wandb/cora/checkpoint_4999.pt").resolve()
    if paths["controller_pretrained_ckpt"] != expected:
        raise ValueError(f"Pilot must start from the original Cora backbone: {expected}")
    with paths["pilot_backbone_args"].open("rb") as stream:
        saved = pickle.load(stream)
    values = deepcopy(vars(saved))
    if values.get("dataset") != "cora":
        raise ValueError("The pilot supports only the saved Cora model")
    name = getattr(cli_args, "name", None)
    if name != f"cora_prespecified_pilot_{target}":
        raise ValueError(f"Pilot run name must be cora_prespecified_pilot_{target}")
    controller_root = _path(getattr(cli_args, "controller_root", None), output=True)
    if controller_root != ROOT / "results/proxy_minimal/controllers":
        raise ValueError("Pilot controller_root must be results/proxy_minimal/controllers")
    log_dir = controller_root / target / name
    audit_path = _path(getattr(cli_args, "pilot_audit_path", None) or log_dir / "pilot_training_audit.json", output=True)
    values.update(PILOT_CONTROLLER_CONFIG)
    values.update({key: str(path) for key, path in paths.items()})
    values.update({
        "prespecified_pilot": True, "operating_point_policy": POLICY,
        "pilot_audit_path": str(audit_path), "controller_root": str(controller_root),
        "device": "cuda:4", "seed": PILOT_TRAIN_SEEDS[target],
        "name": name, "fair_score_metric": target,
        "fair_score_controller_train": True, "fair_score_sp": True,
        "fair_score_learn_k": True, "fair_score_learn_eta": True,
        "parallel": None, "resume": None, "log_tb": False, "log_wandb": False,
        "pilot_automatic_evaluation": False, "pilot_automatic_export": False,
        "pilot_checkpoint_selection": "final_epoch",
    })
    return argparse.Namespace(**values)


def tensor_fingerprint(tensor):
    tensor = tensor.detach().cpu().contiguous()
    # uint8 handles every tensor dtype, including bfloat16 and scalar buffers.
    payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(),
            "dtype": str(tensor.dtype), "shape": list(tensor.shape)}


def snapshot_backbone(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name not in CONTROLLER_KEYS}


def _state_fingerprint(state):
    tensors = {name: tensor_fingerprint(value) for name, value in sorted(state.items())}
    digest = hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "tensor_count": len(tensors), "tensors": tensors}


def verify_backbone(model, before):
    after = snapshot_backbone(model)
    unchanged = set(before) == set(after) and all(
        before[name].dtype == after[name].dtype and torch.equal(before[name], after[name])
        for name in before if name in after)
    return {"backbone_before": _state_fingerprint(before),
            "backbone_after": _state_fingerprint(after),
            "backbone_unchanged": unchanged}


def _assert_frozen(model):
    names = {name for name, param in model.named_parameters() if param.requires_grad}
    if names != CONTROLLER_KEYS:
        raise RuntimeError(f"Only k_t and eta_t may be trainable, got {sorted(names)}")
    if model.training or any(module.training for module in model._denoise_fn.modules()):
        raise RuntimeError("Pilot model and every backbone module must remain in eval mode")
    if any(param.grad is not None for name, param in model.named_parameters() if name not in CONTROLLER_KEYS):
        raise RuntimeError("Frozen backbone received a gradient")


def train_pilot_epochs(args, model, optimizer, audit, persist=lambda: None, log_row=lambda row: None):
    """The existing replay loss unchanged; public for small CPU integrity tests."""
    before = snapshot_backbone(model)
    initial = {name: getattr(model, name).detach().cpu().clone() for name in CONTROLLER_KEYS}
    audit.update({"backbone_before": _state_fingerprint(before),
                  "trainable_names": sorted(name for name, p in model.named_parameters() if p.requires_grad),
                  "controller_diagnostics": {name: {
                      "grad_nonzero_epochs": 0, "grad_missing_epochs": 0, "grad_finite": True,
                      "initial": tensor_fingerprint(initial[name]),
                  } for name in sorted(CONTROLLER_KEYS)}})
    counts = audit["counts"]
    replay = None
    last_loss, last_stats = None, None
    params = [getattr(model, name) for name in sorted(CONTROLLER_KEYS)]
    try:
        _assert_frozen(model)
        persist()
        for epoch in range(args.controller_epochs):
            _assert_frozen(model)
            if epoch % args.controller_replay_refresh == 0:
                # This is controller training replay, never an E1 observation.
                # Release the preceding replay before allocating its replacement.
                replay = None
                counts["replay_sample_calls"] += 1
                with torch.no_grad():
                    replay_graph, replay = model.sample(
                        args.controller_replay_num_samples, return_controller_replay=True)
                    del replay_graph
                counts["replay_graphs"] += args.controller_replay_num_samples
            optimizer.zero_grad(set_to_none=True)
            loss, stats = model.compute_fair_controller_loss_from_replay(replay)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite controller loss at epoch {epoch + 1}; no retry")
            loss.backward()
            _assert_frozen(model)
            grad_stats = {}
            for name in sorted(CONTROLLER_KEYS):
                grad = getattr(model, name).grad
                diagnostics = audit["controller_diagnostics"][name]
                finite = grad is None or bool(torch.isfinite(grad).all())
                diagnostics["grad_finite"] &= finite
                diagnostics["grad_missing_epochs"] += int(grad is None)
                nonzero = int(torch.count_nonzero(grad)) if grad is not None else 0
                diagnostics["grad_nonzero_epochs"] += int(nonzero > 0)
                grad_stats[name] = {"nonzero": nonzero, "finite": finite,
                                    "max_abs": float(grad.detach().abs().max().cpu()) if grad is not None else 0.0}
                if not finite:
                    raise RuntimeError(f"Nonfinite {name} gradient at epoch {epoch + 1}; no retry")
            if args.clip_value is not None:
                torch.nn.utils.clip_grad_value_(params, args.clip_value)
            if args.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, args.clip_norm)
            optimizer.step()
            counts["optimizer_steps"] += 1
            audit["completed_epochs"] = epoch + 1
            last_loss, last_stats = loss.detach(), dict(stats)
            row = {"epoch": epoch + 1, "loss": float(last_loss.cpu()),
                   **last_stats, "controller_gradients": grad_stats}
            log_row(row)
            persist()
    finally:
        audit.update(verify_backbone(model, before))
        for name in sorted(CONTROLLER_KEYS):
            value = getattr(model, name).detach().cpu()
            delta = value - initial[name]
            audit["controller_diagnostics"][name].update({
                "final": tensor_fingerprint(value),
                "param_changed": not torch.equal(value, initial[name]),
                "param_finite": bool(torch.isfinite(value).all()),
                "param_delta_l1": float(delta.abs().sum()),
                "param_delta_max": float(delta.abs().max()),
            })
        persist()
    if not audit["backbone_unchanged"]:
        raise RuntimeError("Backbone tensor changed during controller training")
    if not all(item["param_finite"] for item in audit["controller_diagnostics"].values()):
        raise RuntimeError("Controller parameters became nonfinite; no retry")
    return last_loss, last_stats


def _load_pilot_model(args):
    # Direct preprocessing avoids get_data's real-reference validation/test evaluator.
    from datasets.data_utils import preprocess, EmpiricalEmptyGraphGenerator
    from model import get_model
    with _path(args.pilot_graph).open("rb") as stream:
        graph = pickle.load(stream)
    data = preprocess(graph, degree=args.degree)
    if data.x is None or tuple(data.x.shape) != (graph.number_of_nodes(), args.num_node_feat):
        raise ValueError("Saved backbone features are incompatible with the supplied graph")
    if max(degree for _, degree in graph.degree()) != args.max_degree:
        raise ValueError("Saved backbone max_degree is incompatible with the supplied graph")
    if args.empty_graph_sampler != "empirical" or args.augmented_features:
        raise ValueError("Pilot requires the saved Cora empirical sampler without augmentation")
    sampler = EmpiricalEmptyGraphGenerator([data], degree=args.degree,
                                          augment_features=args.augmented_features)
    model = get_model(args, initial_graph_sampler=sampler).to(args.device)
    checkpoint = torch.load(_path(args.controller_pretrained_ckpt), map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if key not in CONTROLLER_KEYS}
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != CONTROLLER_KEYS or incompatible.unexpected_keys:
        raise ValueError(f"Strict backbone compatibility failed: {incompatible}")
    params = model.freeze_for_fair_controller_training()
    model.eval()
    _assert_frozen(model)
    # Verify copying to the selected device preserved every checkpoint tensor.
    if not verify_backbone(model, state)["backbone_unchanged"]:
        raise RuntimeError("Loaded backbone differs from its original checkpoint")
    return model, params


def run_pilot_training(args):
    """One GPU run, final controller only. Called solely by the opt-in CLI."""
    validate_physical_gpu4(args.device)
    log_dir = _path(Path(args.controller_root) / args.fair_score_metric / args.name, output=True)
    audit_path = _path(args.pilot_audit_path, output=True)
    if log_dir.exists() or audit_path.exists():
        raise FileExistsError(f"Pilot refuses to overwrite/retrain an existing run: {log_dir}")
    audit = {
        "policy": POLICY, "status": "starting", "target": args.fair_score_metric,
        "seed": args.seed, "epochs": args.controller_epochs, "completed_epochs": 0,
        "final_epoch": args.controller_epochs - 1,
        "configuration": dict(PILOT_CONTROLLER_CONFIG),
        "args": vars(args), "checkpoint_selection": "final_epoch",
        "test_used": False, "real_reference_evaluator_constructed": False,
        "automatic_best_reload": False, "hyperparameter_grid": False,
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "assets": {name: file_record(path) for name, path in {
            "backbone_checkpoint": args.controller_pretrained_ckpt,
            "backbone_args": args.pilot_backbone_args, "graph": args.pilot_graph}.items()},
        "counts": {"optimizer_steps": 0, "replay_sample_calls": 0, "replay_graphs": 0,
                   "auto_export_graphs": 0, "evaluation_graphs": 0, "gcn_fits": 0,
                   "retries": 0},
    }
    log_dir.mkdir(parents=True, exist_ok=False)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
        temporary.write_text(json.dumps(audit, indent=2, allow_nan=True) + "\n")
        temporary.replace(audit_path)
    def log_row(row):
        with (log_dir / "controller_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[pilot {args.fair_score_metric}] epoch {row['epoch']}/{args.controller_epochs} loss={row['loss']:.8g}", flush=True)
    start = time.monotonic()
    persist()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 5:
            raise RuntimeError("Physical cuda:4 is unavailable")
        torch.cuda.set_device(4)
        audit["gpu_telemetry"] = start_gpu_telemetry(4)
        audit["gpu_name"] = torch.cuda.get_device_name(4)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.default_generator.manual_seed(args.seed)
        with torch.cuda.device(4):
            torch.cuda.manual_seed(args.seed)
        with (log_dir / "args.pickle").open("wb") as stream:
            pickle.dump(args, stream)
        model, params = _load_pilot_model(args)
        optimizer = torch.optim.Adam(params, lr=args.controller_lr)
        audit["status"] = "training"
        loss, stats = train_pilot_epochs(args, model, optimizer, audit, persist, log_row)
        check_dir = log_dir / "check"
        check_dir.mkdir()
        final_path = check_dir / "controller_final.pt"
        controller = model.get_fair_controller_state_dict()
        controller = {name: value.detach().cpu() if torch.is_tensor(value) else value
                      for name, value in controller.items()}
        torch.save({"controller": controller, "epoch": args.controller_epochs - 1,
                    "stats": stats, "loss": float(loss.cpu()), "args": vars(args),
                    "policy": POLICY, "checkpoint_selection": "final_epoch"}, final_path)
        audit["final_checkpoint"] = file_record(final_path)
        audit["controller_warnings"] = [
            f"{name}: no gradient or parameter change; no retry/search is scheduled"
            for name, item in audit["controller_diagnostics"].items()
            if not item["grad_nonzero_epochs"] or not item["param_changed"]]
        torch.cuda.synchronize(4)
        audit["status"] = "complete"
        print(f"[pilot] Saved explicit final controller: {final_path}", flush=True)
        for warning in audit["controller_warnings"]:
            print(f"[pilot WARNING] {warning}", flush=True)
    except BaseException as exc:
        audit["status"] = "failed"
        audit["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit["wall_seconds"] = time.monotonic() - start
        update_gpu_telemetry(audit.get("gpu_telemetry"))
        persist()
    return audit
