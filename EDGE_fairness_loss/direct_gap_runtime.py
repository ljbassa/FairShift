"""Strict assets and existing-evaluator adapters for read-only gap diagnostics.

No controller calibration or backbone training happens in this module.  A fixed-k
checkpoint must describe its own provenance; legacy learned-k checkpoints are not
silently converted. CPU threading is configured by the CLI before imports.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import types

import numpy as np
import torch


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor):
    tensor = torch.as_tensor(tensor).detach().cpu().contiguous()
    h = hashlib.sha256(str((str(tensor.dtype), tuple(tensor.shape))).encode())
    h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def graph_hash(data):
    from evaluate_generated_graphs import unique_undirected_edge_index
    return object_hash({"num_nodes": int(data.num_nodes),
                        "edges": tensor_hash(unique_undirected_edge_index(data.edge_index))})


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_copy(v) for v in value]
    return deepcopy(value)


def load(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_controller(checkpoint, *, backbone_hash, timesteps, dataset):
    """Require an explicit fixed-k, eta-only checkpoint, not just constant logits."""
    if checkpoint.get("format") != "fairshift_fixed_k_eta_v1":
        raise ValueError("Compatible fixed-k/eta-only checkpoint mapping is missing; "
                         "legacy learned-k checkpoints cannot be converted by observe")
    if checkpoint.get("backbone_hash") != backbone_hash or checkpoint.get("dataset") != dataset:
        raise ValueError("Controller backbone hash/dataset mismatch")
    if checkpoint.get("num_timesteps") != timesteps:
        raise ValueError("Controller/backbone T mismatch")
    k = checkpoint.get("fixed_k")
    if not isinstance(k, (int, float)) or not np.isfinite(k) or not 0 < k <= 1:
        raise ValueError("fixed_k must be finite and in (0,1]")
    variant = checkpoint.get("variant")
    if variant not in ("T", "tied"):
        raise ValueError("Calibrated controller variant must be T or tied")
    expected = ["eta_t"] if variant == "T" else ["eta"]
    if checkpoint.get("trainable_parameters") != expected:
        raise ValueError("Only eta may have been calibrated")
    if checkpoint.get("tracking_loss_weight") != 0 or checkpoint.get("learned_k") is not False:
        raise ValueError("Learned k/tracking is outside the fixed-k diagnostic contract")
    if checkpoint.get("uses_test_for_selection") is not False:
        raise ValueError("Test-based checkpoint selection is prohibited")
    for key in ("calibration_seed", "checkpoint_rule", "objective", "budget", "code_revision",
                "fair_label_attr", "eo_min_mass", "backbone_args_hash", "reference_graph_hash",
                "feature_hash", "group_hash"):
        if key not in checkpoint or checkpoint[key] is None:
            raise ValueError(f"Controller lacks required provenance: {key}")
    eta = torch.as_tensor(checkpoint.get("eta_schedule"), dtype=torch.float32).reshape(-1)
    if eta.numel() != timesteps or not torch.isfinite(eta).all() or (eta <= 0).any():
        raise ValueError("eta_schedule must have exactly T finite positive values in reverse-t order")
    if variant == "tied" and not torch.equal(eta, eta[0].expand_as(eta)):
        raise ValueError("Tied eta checkpoint has a nonconstant schedule")
    if checkpoint.get("metric") not in ("sp", "eo") or type(checkpoint.get("normalization")) is not bool:
        raise ValueError("Controller metric and normalization must be explicit")
    if not isinstance(checkpoint["fair_label_attr"], str) or not checkpoint["fair_label_attr"]:
        raise ValueError("Controller group source is missing")
    if not np.isfinite(checkpoint["eo_min_mass"]) or checkpoint["eo_min_mass"] < 0:
        raise ValueError("Invalid EO condition threshold")
    return {**checkpoint, "eta_schedule": eta.tolist()}


def attach_fixed_schedule(model, k, eta, *, train_eta=False, tied=False):
    """Select fixed schedules on the existing controller sampler path.

    This adapter is explicit in the resolved manifest. No cache, candidate,
    normalization or guidance formula is replaced. k=1 is exact, not sigmoid(large).
    """
    if train_eta:
        raise ValueError("Observe adapter does not calibrate; use explicit ablation calibration command")
    eta = torch.as_tensor(eta, dtype=torch.float32, device=model.device).reshape(-1)
    if eta.numel() != model.num_timesteps or (eta <= 0).any() or not torch.isfinite(eta).all():
        raise ValueError("Invalid eta schedule")
    if not 0 < float(k) <= 1:
        raise ValueError("Invalid fixed k")
    # Native fixed_one has no k parameter; native shared eta has shape [1].
    # Preserve both native representations while the explicit adapter supplies
    # the full observed schedules independently of their storage shapes.
    for name in ("fair_score_k_raw", "fair_score_eta_raw"):
        parameter = getattr(model, name, None)
        if parameter is not None:
            parameter.requires_grad_(False)
    model.register_buffer("_direct_gap_fixed_k", torch.full_like(eta, float(k)))
    model.register_buffer("_direct_gap_eta", eta.clone())

    def fixed_k(self, *args, **kwargs):
        values = self._direct_gap_fixed_k
        indices = kwargs.get("t_graph")
        return values if indices is None else values[indices.to(values.device).long()]

    def fixed_eta(self, *args, **kwargs):
        values = self._direct_gap_eta
        indices = kwargs.get("t_graph")
        return values if indices is None else values[indices.to(values.device).long()]

    model._get_effective_fair_score_k = types.MethodType(fixed_k, model)
    model._get_effective_fair_score_eta = types.MethodType(fixed_eta, model)
    return model


def build_observed_model(config, *, device):
    from datasets.data_utils import preprocess, EmpiricalEmptyGraphGenerator
    from model import get_model
    from evaluate_generated_graphs import get_lp_group_vector, get_local_attr_vector

    with Path(config["backbone_args"]).open("rb") as stream:
        args = pickle.load(stream)
    if args.dataset != config["dataset"]:
        raise ValueError("Backbone args dataset mismatch")
    with Path(config["graph"]).open("rb") as stream:
        reference = pickle.load(stream)
    data = preprocess(reference, degree=args.degree)
    if data.x is None or data.x.shape[1] != args.num_node_feat:
        raise ValueError("Actual features are incompatible with backbone")
    if max(d for _, d in reference.degree()) != args.max_degree:
        raise ValueError("Reference graph max_degree mismatch")
    if args.empty_graph_sampler != "empirical" or args.augmented_features:
        raise ValueError("Only audited empirical sampler without augmentation is supported")
    sampler = EmpiricalEmptyGraphGenerator([data], degree=args.degree,
                                           augment_features=args.augmented_features)
    backbone_digest = file_hash(config["backbone_checkpoint"])
    controller = config["controller"]
    if controller["kind"] in ("T", "tied"):
        resolved = validate_controller(load(controller["checkpoint"]), backbone_hash=backbone_digest,
                                       timesteps=args.diffusion_steps, dataset=args.dataset)
        if resolved["variant"] != controller["kind"]:
            raise ValueError("Controller variant mismatch")
        for key, expected in (("backbone_args_hash", file_hash(config["backbone_args"])),
                              ("reference_graph_hash", file_hash(config["graph"])),
                              ("feature_hash", tensor_hash(data.x))):
            if resolved[key] != expected:
                raise ValueError(f"Controller asset semantics mismatch: {key}")
        args.fair_label_attr = resolved["fair_label_attr"]
        args.fair_score_eo_min_mass = resolved["eo_min_mass"]
        controller_digest = file_hash(controller["checkpoint"])
    elif controller["kind"] == "F":
        # A new explicit F configuration is allowed, but never inferred from T.
        if controller.get("uses_test_for_selection") is not False:
            raise ValueError("F config must predeclare no test selection")
        for key in ("k", "eta", "metric", "normalization", "configuration_source"):
            if key not in controller:
                raise ValueError(f"Missing explicit F setting {key}")
        if type(controller["normalization"]) is not bool or controller["metric"] not in ("sp", "eo"):
            raise ValueError("F normalization must be a bool and metric must be sp/eo")
        for key in ("k", "eta"):
            if type(controller[key]) not in (int, float) or not np.isfinite(controller[key]) or controller[key] <= 0:
                raise ValueError(f"F {key} must be finite positive")
        if controller["k"] > 1:
            raise ValueError("F k must be in (0,1]")
        resolved = {"variant": "F", "fixed_k": controller["k"],
                    "eta_schedule": [controller["eta"]] * args.diffusion_steps,
                    "metric": controller["metric"], "normalization": controller["normalization"],
                    "calibration_seed": None, "checkpoint_rule": "fixed_configuration",
                    "configuration_source": controller["configuration_source"]}
        args.fair_label_attr = controller.get("fair_label_attr", getattr(args, "fair_label_attr", "y"))
        args.fair_score_eo_min_mass = controller.get("eo_min_mass", getattr(args, "fair_score_eo_min_mass", 1e-6))
        controller_digest = object_hash(controller)
    else:
        raise ValueError("Observe supports explicit fixed-k F/T/tied only")
    if resolved["metric"] != config["target"]:
        raise ValueError("Target and checkpoint metric mismatch")
    sampled_groups = get_local_attr_vector(data, args.fair_label_attr).long()
    evaluated_groups = get_lp_group_vector(data, preferred_attr=config.get("group_attr", "y")).long()
    if not torch.equal(sampled_groups, evaluated_groups):
        raise ValueError("Sampler and existing evaluator use different group metadata; explicit compatible mapping required")
    if "group_hash" in resolved and resolved["group_hash"] != tensor_hash(sampled_groups):
        raise ValueError("Controller calibrated with different group metadata")
    args.device = device
    args.fair_score_controller_train = True
    args.fair_score_metric = resolved["metric"]
    args.fair_score_guidance_normalize = resolved["normalization"]
    args.fair_score_k = resolved["fixed_k"]
    args.fair_score_eta = resolved["eta_schedule"][0]
    args.fair_score_k_tracking_loss_weight = 0.0
    model = get_model(args, sampler)
    checkpoint = load(config["backbone_checkpoint"])
    state = checkpoint.get("model", checkpoint)
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if key not in
             ("fair_score_k_raw", "fair_score_eta_raw")}
    expected_controller_keys = {key for key in ("fair_score_k_raw", "fair_score_eta_raw")
                                if key in model.state_dict()}
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != expected_controller_keys or incompatible.unexpected_keys:
        raise ValueError(f"Strict backbone tensor key/shape compatibility failed: {incompatible}")
    model.to(device).eval().requires_grad_(False)
    attach_fixed_schedule(model, resolved["fixed_k"], resolved["eta_schedule"])
    actual = {"T": model.num_timesteps, "k": resolved["fixed_k"],
              "eta_schedule": resolved["eta_schedule"], "schedule_index": "reverse_t_ascending; t=0 final",
              "normalization": model.fair_score_guidance_normalize,
              "fair_label_attr": model.fair_label_attr, "eo_min_mass": model.fair_score_eo_min_mass,
              "variant": resolved["variant"], "controller": resolved,
              "backbone_hash": backbone_digest, "controller_hash": controller_digest,
              "args": vars(args), "generation_mode": "online_guided_no_replay"}
    return model, data, actual


@contextmanager
def evaluator_device(device):
    """Sequential runner device adapter; original default evaluator stays intact."""
    import evaluate_generated_graphs as evaluator
    previous = evaluator.samplepy_device
    evaluator.samplepy_device = lambda: torch.device(device)
    try:
        yield evaluator
    finally:
        evaluator.samplepy_device = previous


def fit_existing_evaluator(data, *, split_seed, evaluator_seed, device="cpu", group_attr="y"):
    """Call the existing split, 48-config tuning, fit/early-stop, and predict code.

    Saves the actually selected model and masks. There is no extra full-U scoring
    and no GNN fit for intermediate snapshots. Distinct seed boundaries are explicit.
    """
    from proxy_minimal_gcn import seed_all
    with evaluator_device(device) as old:
        data = old.ensure_features(data)
        seed_all(int(split_seed), "cpu")
        split = old.samplepy_prepare_for_gae(data)
        groups = old.get_lp_group_vector(data, preferred_attr=group_attr).long()
        group_source = "sens" if getattr(data, "sens", None) is not None else group_attr
        if getattr(data, group_source, None) is None:
            group_source = "y"
        adjacency, features, labels = old.samplepy_preprocess(
            split["A_train"], split["A_full"], data.x.detach().cpu().float(), groups, None, None)
        seed_all(int(evaluator_seed), device)
        best_auc, best_model, best_meta = -1.0, None, {}
        trials = []
        for config in old.samplepy_config_list():
            auc, _, _, model, meta = old.samplepy_fit_trial(
                adjacency, features, groups, labels, split["train_mask"], split["val_mask"], **config)
            trials.append({"config": config, "validation": meta})
            if np.isfinite(auc) and auc > best_auc:
                best_auc, best_model, best_meta = float(auc), model, meta
            if auc == 1.0:
                break
        if best_model is None:
            raise RuntimeError("Existing evaluator selected no finite validation model; no retry")
        with torch.no_grad():
            auc, _, _, raw = old.samplepy_predict(
                adjacency, features, groups, labels, split["test_mask"], best_model)
        pair_ids = split["test_mask"].nonzero().t().contiguous()
        result = {"pair_ids": pair_ids, "scores": raw["scores"], "labels": raw["labels"],
                  "node_groups": groups, "num_nodes": int(data.num_nodes),
                  "node_batch": torch.zeros(data.num_nodes, dtype=torch.long),
                  "generated_positive_pairs": old.unique_undirected_edge_index(data.edge_index),
                  "auc": float(auc), "train_pairs": split["train_mask"].nonzero().t().contiguous(),
                  "val_pairs": split["val_mask"].nonzero().t().contiguous(),
                  "val_labels": labels[split["val_mask"].to(device)].detach().cpu(),
                  "selected_model_state": cpu_copy(best_model.state_dict()),
                  "selected_model_meta": best_meta, "trials": trials,
                  "split_seed": int(split_seed), "evaluator_seed": int(evaluator_seed),
                  "feature_source": "generated_graph.x (existing node metadata)",
                  "feature_hash": tensor_hash(data.x), "group_source": f"generated_graph.{group_source}",
                  "group_hash": tensor_hash(groups), "protocol": "existing_samplepy_grid_v1",
                  "split_rule": "existing generated positives 80/10/10; disjoint true-nonedge val/test",
                  "training_negative_rule": "existing uniform ordered pairs including possible collisions",
                  "selection_rule": "validation AUC grid + patience=5; no test selection"}
        return cpu_copy(result)
