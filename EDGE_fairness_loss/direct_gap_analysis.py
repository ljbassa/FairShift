"""Offline signed cache/evaluator gap diagnostics; no model or decoder is called.

``analyze_snapshot`` joins the entire saved cache U to actual generated-test
pairs P by canonical pair IDs.  SP always uses *all* P, and EO uses generated
held-out positives.  Inputs carry independently saved provenance; callers must
never attach a new graph/checkpoint identity to an unrelated legacy snapshot.

Scores and differences use score units.  Only fields ending in ``_pp`` are
percentage points.  Invalid statistics remain NaN with explicit reasons.
"""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SIGN_THRESHOLD = 1e-6  # Prespecified in score units, not estimated from results.
IDENTITY_KEYS = (
    "graph_id", "graph_hash", "backbone_hash", "controller_hash",
    "node_order_hash", "pair_mapping", "batch_id",
)
GROUP_KEYS = ("dataset", "phase", "score_name", "progress", "chunk_index", "protocol", "metric")
NAN = float("nan")


def _array(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _vector(value, size, name, dtype=None):
    result = _array(value, dtype)
    if result.ndim != 1 or len(result) != size:
        raise ValueError(f"{name} must have shape [{size}]")
    return result


def _pairs(value, num_nodes, name):
    original = _array(value)
    if original.ndim != 2 or original.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2, E]")
    if not np.issubdtype(original.dtype, np.integer):
        raise ValueError(f"{name} must contain integer node IDs")
    pairs = original.astype(np.int64, copy=False)
    if np.any(pairs < 0) or np.any(pairs >= num_nodes):
        raise ValueError(f"{name} has out-of-range node IDs")
    if np.any(pairs[0] >= pairs[1]):
        raise ValueError(f"{name} must use i<j; self-loops and reversed IDs are forbidden")
    codes = pairs[0] * np.int64(num_nodes) + pairs[1]
    if len(np.unique(codes)) != len(codes):
        raise ValueError(f"{name} contains duplicate unordered pairs")
    return pairs, codes


def _metadata(mapping):
    meta = dict(mapping.get("provenance", {}))
    for key, value in mapping.items():
        if key != "provenance" and isinstance(value, (str, int, float, bool, type(None))):
            if key in meta and meta[key] != value:
                raise ValueError(f"Conflicting flat/provenance field: {key}")
            meta[key] = value
    return meta


def _validate_provenance(snapshot, evaluation, provenance):
    saved, evaluated = _metadata(snapshot), _metadata(evaluation)
    expected = dict(provenance or {})
    for key in IDENTITY_KEYS:
        if key not in saved or key not in evaluated:
            raise ValueError(f"Missing independently bound provenance: {key}")
        if saved[key] != evaluated[key]:
            raise ValueError(f"Snapshot/evaluator provenance mismatch: {key}")
        if key in expected and saved[key] != expected[key]:
            raise ValueError(f"Requested provenance mismatch: {key}")
    if saved["pair_mapping"] != "unordered_i_lt_j":
        raise ValueError("pair_mapping must be unordered_i_lt_j")
    phase = saved.get("phase", saved.get("snapshot_phase"))
    if phase not in ("pre", "post"):
        raise ValueError("Snapshot phase must explicitly be pre or post")
    score_name = saved.get("score_name")
    if score_name != {"pre": "q_bar", "post": "q_final"}[phase]:
        raise ValueError("Snapshot phase/score_name mismatch")
    for source in (saved, evaluated, expected):
        for key in ("phase", "snapshot_phase"):
            if key in source and source[key] != phase:
                raise ValueError(f"Snapshot phase provenance mismatch: {key}")
    # These identify acquisition/evaluation protocols; do not silently overwrite
    # a saved value with a manifest value just because the hashes match.
    for key in ("dataset", "configuration_id", "controller_seed", "graph_seed", "evaluator_seed",
                "split_seed", "evaluator_checkpoint_hash", "split_hash", "protocol",
                "feature_source", "group_source", "snapshot_id"):
        values = [source[key] for source in (saved, evaluated, expected) if key in source]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"Provenance mismatch: {key}")
    result = {**saved, **evaluated, **expected, "phase": phase, "snapshot_phase": phase,
              "score_name": score_name}
    progress = result.get("progress")
    if progress is None or not np.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError("Snapshot progress must be finite and between zero and one")
    if phase == "post" and progress != 1.0:
        raise ValueError("q_final requires final post-shift progress=1")
    result.pop("num_nodes", None)
    return result


def _gap(score, same, prefix, *, weights=None, min_mass=1e-6):
    """Group sums/counts/means; missing/invalid groups never become zero gaps."""
    result, reasons, means = {}, [], []
    weights = np.ones(len(score), dtype=np.float64) if weights is None else weights
    for name, mask in (("same", same), ("different", ~same)):
        v, w = score[mask], weights[mask]
        positive = w > 0
        reason = ""
        if len(v) == 0:
            reason = "empty_group"
        elif not np.isfinite(w).all() or np.any(w < 0):
            reason = "invalid_weight"
        elif not np.isfinite(v[positive]).all():
            reason = "nonfinite_score"
        mass = float(w.sum())
        if not reason and mass <= min_mass:
            reason = "insufficient_weight_mass"
        # Zero-weight scores do not enter the weighted statistic, including NaN.
        weighted_sum = float(np.sum(v[positive] * w[positive])) if not reason else NAN
        mean = weighted_sum / mass if not reason else NAN
        result.update({f"{prefix}_{name}_count": int(len(v)),
                       f"{prefix}_{name}_positive_weight_count": int(positive.sum()),
                       f"{prefix}_{name}_finite_count": int(np.isfinite(v).sum()),
                       f"{prefix}_{name}_sum": float(v.sum()) if np.isfinite(v).all() else NAN,
                       f"{prefix}_{name}_weighted_sum": weighted_sum,
                       f"{prefix}_{name}_weight_mass": mass,
                       f"{prefix}_{name}_mean": mean,
                       f"{prefix}_{name}_invalid_reason": reason})
        means.append(mean)
        if reason:
            reasons.append(f"{name}:{reason}")
    value = float(means[0] - means[1]) if not reasons else NAN
    result.update({prefix: value, f"{prefix}_abs": abs(value), f"{prefix}_pp": 100.0 * value,
                   f"{prefix}_abs_pp": 100.0 * abs(value), f"{prefix}_valid": not reasons,
                   f"{prefix}_invalid_reason": ";".join(reasons)})
    return result


def _rankdata(values):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
    return ranks


def safe_correlation(x, y, method="pearson"):
    """Finite-pair association with an explicit undefined reason and counts."""
    x, y = _array(x, np.float64), _array(y, np.float64)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Correlation inputs must be equally sized vectors")
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be pearson or spearman")
    valid = np.isfinite(x) & np.isfinite(y)
    result = {"value": NAN, "valid": False, "pair_count": int(valid.sum()),
              "total_pair_count": int(len(x)), "excluded_nonfinite_count": int((~valid).sum()),
              "invalid_reason": ""}
    x, y = x[valid], y[valid]
    if len(x) < 2:
        result["invalid_reason"] = "insufficient_finite_pairs"
    elif np.ptp(x) == 0 or np.ptp(y) == 0:
        result["invalid_reason"] = "constant_input"
    else:
        if method == "spearman":
            x, y = _rankdata(x), _rankdata(y)
        value = float(np.corrcoef(x, y)[0, 1])
        result.update(value=value, valid=bool(np.isfinite(value)),
                      invalid_reason="" if np.isfinite(value) else "nonfinite_correlation")
    return result


def _auc(labels, scores):
    if not np.isfinite(scores).all():
        return NAN, "nonfinite_score"
    pos, neg = int(labels.sum()), int(len(labels) - labels.sum())
    if not pos or not neg:
        return NAN, "missing_label_class"
    ranks = _rankdata(scores)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg)), ""


def _differences(row, terms, downstream, sign_threshold):
    for name, left, right in terms:
        value = float(row[left] - row[right])
        row.update({name: value, f"{name}_pp": 100 * value, f"{name}_abs": abs(value),
                    f"{name}_abs_pp": 100 * abs(value)})
    total = float(row["a"] - row[downstream])
    summands = [row[name] for name, _, _ in terms]
    valid = bool(np.isfinite([total, *summands]).all())
    residual = total - sum(summands) if valid else NAN
    if valid and not np.isclose(total, sum(summands), atol=1e-12, rtol=1e-10):
        raise ArithmeticError("Signed gap decomposition identity failed")
    row.update(total_difference=total, total_difference_pp=100 * total,
               total_difference_abs=abs(total), total_difference_abs_pp=100 * abs(total),
               identity_residual=residual, identity_valid=valid,
               identity_invalid_reason="" if valid else "undefined_gap_term",
               downstream_gap=row[downstream], downstream_abs_gap=abs(row[downstream]),
               downstream_gap_pp=100 * row[downstream],
               downstream_abs_gap_pp=100 * abs(row[downstream]),
               abs_a_minus_c=abs(row["a"] - row["c"]),
               abs_b_minus_c=abs(row["b"] - row["c"]))
    row["abs_a_minus_c_pp"] = row["abs_a_minus_c"] * 100
    row["abs_b_minus_c_pp"] = row["abs_b_minus_c"] * 100
    for key in ("a", "b", "c", "d"):
        if key in row:
            row[f"{key}_sign_valid"] = bool(np.isfinite(row[key]) and abs(row[key]) > sign_threshold)
    for key in ("a", "b"):
        valid_sign = row[f"{key}_sign_valid"] and row[f"{downstream}_sign_valid"]
        row[f"{key}_downstream_sign_valid"] = bool(valid_sign)
        row[f"{key}_downstream_sign_agreement"] = (
            float(np.sign(row[key]) == np.sign(row[downstream])) if valid_sign else NAN)


def analyze_snapshot(snapshot, evaluation, provenance=None, *, min_mass=1e-6,
                     sign_threshold=SIGN_THRESHOLD):
    """Return SP and EO flat records from one immutable snapshot and evaluator.

    Evaluation requires ``pair_ids, scores, labels, generated_positive_pairs,
    node_groups, num_nodes``.  Both inputs require ``provenance`` containing
    IDENTITY_KEYS (flat metadata is also accepted).  Snapshot ``phase`` is
    ``pre``/``post`` and ``score_name`` is ``q_bar``/``q_final`` respectively.
    An EO row is explicitly unavailable when no actual condition cache w exists.
    """
    if not np.isfinite(min_mass) or min_mass < 0 or not np.isfinite(sign_threshold) or sign_threshold < 0:
        raise ValueError("Mass and sign thresholds must be finite and nonnegative")
    meta = _validate_provenance(snapshot, evaluation, provenance)
    num_nodes = int(evaluation["num_nodes"])
    if num_nodes < 1 or num_nodes > 3_037_000_499:
        raise ValueError("num_nodes is invalid for int64 pair IDs")
    u, u_codes = _pairs(snapshot["pair_ids"], num_nodes, "cache pair_ids")
    p, p_codes = _pairs(evaluation["pair_ids"], num_nodes, "evaluation pair_ids")
    positive_pairs, positive_codes = _pairs(evaluation["generated_positive_pairs"], num_nodes,
                                            "generated_positive_pairs")
    q = _vector(snapshot["q"], len(u_codes), "q", np.float64)
    scores = _vector(evaluation["scores"], len(p_codes), "scores", np.float64)
    labels = _vector(evaluation["labels"], len(p_codes), "labels", np.float64)
    groups = _vector(evaluation["node_groups"], num_nodes, "node_groups")
    if any(value is None for value in groups) or (
            np.issubdtype(groups.dtype, np.number) and not np.isfinite(groups).all()):
        raise ValueError("Missing/nonfinite node group metadata")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("Evaluation labels must be binary")
    if not np.array_equal(labels.astype(bool), np.isin(p_codes, positive_codes)):
        raise ValueError("Evaluation labels do not match the completed generated graph")
    node_batch_value = evaluation.get("node_batch", snapshot.get("node_batch"))
    if node_batch_value is not None:
        node_batch = _vector(node_batch_value, num_nodes, "node_batch")
        if "node_batch" in snapshot and not np.array_equal(node_batch, _array(snapshot["node_batch"])):
            raise ValueError("Snapshot/evaluator node_batch mismatch")
        for pairs, name in ((u, "cache"), (p, "evaluation"), (positive_pairs, "generated graph")):
            if np.any(node_batch[pairs[0]] != node_batch[pairs[1]]):
                raise ValueError(f"{name} contains cross-batch pairs")
            if np.any(node_batch[pairs[0]] != meta["batch_id"]):
                raise ValueError(f"{name} contains pairs from a different graph batch")
    elif meta["batch_id"] != 0:
        raise ValueError("Nonzero batch_id requires explicit node_batch")
    if "pair_batch" in snapshot:
        pair_batch = _vector(snapshot["pair_batch"], len(u_codes), "pair_batch")
        if np.any(pair_batch != meta["batch_id"]):
            raise ValueError("Cache pair_batch contains a different graph batch")
    same_u, same_p = groups[u[0]] == groups[u[1]], groups[p[0]] == groups[p[1]]
    if "same_mask" in snapshot and not np.array_equal(
            _vector(snapshot["same_mask"], len(u_codes), "same_mask", bool), same_u):
        raise ValueError("Cached same_mask disagrees with node group metadata")
    order = np.argsort(u_codes)
    if not np.isin(positive_codes, u_codes).all():
        raise ValueError("Generated positive pair is missing from the cache (full U)")
    positions = np.searchsorted(u_codes[order], p_codes)
    if not len(u_codes) and len(p_codes):
        raise ValueError("Evaluation pair is missing from the cache")
    if len(p_codes) and (np.any(positions >= len(u_codes)) or
                         not np.array_equal(u_codes[order[positions]], p_codes)):
        raise ValueError("Evaluation pair is missing from the cache")
    aligned_q = q[order[positions]]
    auc, auc_reason = _auc(labels, scores)
    positive_mask = labels.astype(bool)
    final_eo = _gap(scores[positive_mask], same_p[positive_mask], "gnn_eo", min_mass=min_mass)
    graph_node_count = (int(np.count_nonzero(node_batch == meta["batch_id"]))
                        if node_batch_value is not None else num_nodes)
    possible = graph_node_count * (graph_node_count - 1) // 2
    base = {**meta, "num_nodes": num_nodes, "score_unit": "probability", "pp_multiplier": 100,
            "sign_threshold": float(sign_threshold), "min_mass": float(min_mass),
            "graph_node_count": graph_node_count,
            "cache_pair_count": len(u_codes), "evaluation_pair_count": len(p_codes),
            "evaluation_positive_count": int(positive_mask.sum()),
            "generated_edge_count": len(positive_codes),
            "cache_coverage": len(u_codes) / possible if possible else NAN,
            "evaluation_cache_coverage": 1.0 if len(p_codes) else NAN,
            "auc": auc, "auc_valid": not auc_reason, "auc_invalid_reason": auc_reason,
            "chunk_index": meta.get("chunk_index", 0), **final_eo}
    if "visited_mask" in snapshot:
        visited = _vector(snapshot["visited_mask"], len(u_codes), "visited_mask", bool)
        base.update(cache_visited_count=int(visited.sum()), cache_unvisited_count=int((~visited).sum()))
    sp = {**base, "metric": "sp", "diagnostic_status": "available", "unavailable_reason": ""}
    for name, values, same in (("a", q, same_u), ("b", aligned_q, same_p), ("c", scores, same_p)):
        sp.update(_gap(values, same, name, min_mass=min_mass))
    _differences(sp, (("support_difference", "a", "b"), ("score_difference", "b", "c")),
                 "c", sign_threshold)
    eo = {**base, "metric": "eo", "diagnostic_status": "available", "unavailable_reason": ""}
    eo.update(_gap(aligned_q[positive_mask], same_p[positive_mask], "c", min_mass=min_mass))
    eo.update(_gap(scores[positive_mask], same_p[positive_mask], "d", min_mass=min_mass))
    if snapshot.get("w") is None:
        eo.update(diagnostic_status="unavailable", unavailable_reason="actual_condition_cache_w_not_saved")
        for name in ("a", "b"):
            eo.update({name: NAN, f"{name}_abs": NAN, f"{name}_pp": NAN, f"{name}_abs_pp": NAN,
                       f"{name}_valid": False, f"{name}_invalid_reason": "actual_condition_cache_w_not_saved"})
    else:
        w = _vector(snapshot["w"], len(u_codes), "w", np.float64)
        eo.update(_gap(q, same_u, "a", weights=w, min_mass=min_mass))
        eo.update(_gap(aligned_q, same_p, "b", weights=w[order[positions]], min_mass=min_mass))
    _differences(eo, (("support_difference", "a", "b"),
                      ("conditioning_difference", "b", "c"), ("score_difference", "c", "d")),
                 "d", sign_threshold)
    return [sp, eo]


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else NAN


def _average_unit(rows):
    """Invalid repeats invalidate the unit, rather than silently being selected out."""
    keys = ("a", "b", "c", "d", "downstream_gap", "support_difference", "conditioning_difference",
            "score_difference", "total_difference", "total_difference_abs", "score_difference_abs",
            "abs_a_minus_c", "abs_b_minus_c", "auc")
    unit = {key: _finite_mean([row.get(key, NAN) for row in rows]) for key in keys}
    unit.update(diagnostic_status="available" if all(row.get("diagnostic_status") == "available" for row in rows)
                else "unavailable", source_repeat_count=len(rows))
    return unit


def _hierarchical_units(rows):
    graphs = defaultdict(list)
    for row in rows:
        for key in ("dataset", "configuration_id", "controller_seed", "graph_seed", "evaluator_seed",
                    "graph_id", "graph_hash", "protocol"):
            if key not in row:
                raise ValueError(f"Aggregation requires hierarchy field: {key}")
        graph_key = tuple(row[key] for key in ("configuration_id", "controller_seed", "graph_seed", "graph_id", "graph_hash"))
        graphs[graph_key].append(row)
    graph_units = []
    for key, repeats in graphs.items():
        seeds = [row["evaluator_seed"] for row in repeats]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate graph/evaluator observation in one snapshot group; do not pool snapshots")
        identities = {(row.get("backbone_hash"), row.get("controller_hash")) for row in repeats}
        if len(identities) != 1:
            raise ValueError("A graph's evaluator repeats have different checkpoint provenance")
        snapshot_ids = {row.get("snapshot_id") for row in repeats}
        if len(snapshot_ids) != 1:
            raise ValueError("A graph's evaluator repeats must refer to the same snapshot")
        graph_units.append({**_average_unit(repeats), "configuration_id": key[0], "controller_seed": key[1],
                            "graph_seed": key[2], "graph_id": key[3], "graph_hash": key[4],
                            "evaluator_repeat_count": len(repeats), "graph_count": 1})
    controllers = defaultdict(list)
    for unit in graph_units:
        controllers[(unit["configuration_id"], unit["controller_seed"])].append(unit)
    controller_units = []
    for key, graphs_for_controller in controllers.items():
        controller_units.append({**_average_unit(graphs_for_controller), "configuration_id": key[0],
                                 "controller_seed": key[1], "graph_count": len(graphs_for_controller),
                                 "evaluator_repeat_count": sum(g["evaluator_repeat_count"] for g in graphs_for_controller)})
    configs = defaultdict(list)
    for unit in controller_units:
        configs[unit["configuration_id"]].append(unit)
    config_units = []
    for key, config_controllers in configs.items():
        config_units.append({**_average_unit(config_controllers), "configuration_id": key,
                             "controller_count": len(config_controllers),
                             "graph_count": sum(c["graph_count"] for c in config_controllers),
                             "evaluator_repeat_count": sum(c["evaluator_repeat_count"] for c in config_controllers)})
    return graph_units, config_units, len(controllers)


def _group_records(records):
    groups = defaultdict(list)
    for row in records:
        for key in GROUP_KEYS:
            if key not in row:
                raise ValueError(f"Record is missing grouping field: {key}")
        groups[tuple(row[key] for key in GROUP_KEYS)].append(row)
    # Present final post-shift first, without selecting favorable observations.
    return sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] != "post", str(item[0])))


def summarize_records(records, *, sign_threshold=SIGN_THRESHOLD):
    """Association by dataset/phase/progress/protocol, after averaging GNN repeats.

    Graph units average evaluator seeds.  Configuration units average graphs
    within each controller seed, then give controller seeds equal weight.
    No snapshot phases/progresses are pooled.  Nonfinite units stay visible.
    """
    if not np.isfinite(sign_threshold) or sign_threshold < 0:
        raise ValueError("sign_threshold must be finite and nonnegative")
    summaries = []
    for group, rows in _group_records(records):
        if any(float(row.get("sign_threshold", sign_threshold)) != sign_threshold for row in rows):
            raise ValueError("Record sign threshold differs from prespecified aggregation threshold")
        graph_units, config_units, controller_count = _hierarchical_units(rows)
        for level, units in (("graph", graph_units), ("configuration_mean", config_units)):
            result = {**dict(zip(GROUP_KEYS, group)), "analysis_level": level, "unit_count": len(units),
                      "record_count": len(rows), "graph_count": len(graph_units),
                      "configuration_count": len(config_units), "controller_count": controller_count,
                      "evaluator_repeat_count": sum(unit["evaluator_repeat_count"] for unit in graph_units),
                      "evaluator_fit_count": sum(bool(row.get("evaluator_fit_completed", True)) for row in rows),
                      "unavailable_record_count": sum(row.get("diagnostic_status") != "available" for row in rows),
                      "sign_threshold": sign_threshold,
                      "configuration_weighting": "equal_controller_seeds_after_equal_graphs_after_equal_evaluators",
                      "interpretation": "Empirical alignment; no causal downstream descent guarantee"}
            target = np.asarray([unit["downstream_gap"] for unit in units], dtype=np.float64)
            for predictor in ("a", "b"):
                values = np.asarray([unit[predictor] for unit in units], dtype=np.float64)
                for absolute in (False, True):
                    x, y = (np.abs(values), np.abs(target)) if absolute else (values, target)
                    prefix = f"{'absolute' if absolute else 'signed'}_{predictor}_downstream"
                    for method in ("pearson", "spearman"):
                        for key, value in safe_correlation(x, y, method).items():
                            result[f"{prefix}_{method}_{key}"] = value
                sign_valid = (np.isfinite(values) & np.isfinite(target) &
                              (np.abs(values) > sign_threshold) & (np.abs(target) > sign_threshold))
                result[f"{predictor}_sign_valid_count"] = int(sign_valid.sum())
                result[f"{predictor}_sign_excluded_count"] = int((~sign_valid).sum())
                result[f"{predictor}_sign_agreement"] = (
                    float(np.mean(np.sign(values[sign_valid]) == np.sign(target[sign_valid])))
                    if sign_valid.any() else NAN)
                result[f"{predictor}_sign_invalid_reason"] = "" if sign_valid.any() else "no_nonzero_finite_pairs"
                finite = np.isfinite(values) & np.isfinite(target)
                result[f"{predictor}_downstream_abs_discrepancy_mean"] = (
                    float(np.mean(np.abs(values[finite] - target[finite]))) if finite.any() else NAN)
                result[f"{predictor}_downstream_discrepancy_valid_count"] = int(finite.sum())
            for key in ("a", "b", "c", "d", "downstream_gap", "support_difference", "conditioning_difference",
                        "score_difference", "total_difference", "total_difference_abs", "score_difference_abs", "auc"):
                values = np.asarray([unit.get(key, NAN) for unit in units], dtype=np.float64)
                finite = values[np.isfinite(values)]
                result[f"{key}_mean"] = float(finite.mean()) if len(finite) else NAN
                result[f"{key}_std"] = float(finite.std(ddof=1)) if len(finite) >= 2 else NAN
                result[f"{key}_valid_count"] = len(finite)
            summaries.append(result)
    return summaries


def _write_csv(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_groups(records, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for group, rows in _group_records(records):
        graph_units, config_units, _ = _hierarchical_units(rows)
        label = dict(zip(GROUP_KEYS, group))
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(str(value) for value in group))[:140]
        suffix = hashlib.sha256(json.dumps(group, default=str).encode()).hexdigest()[:10]
        for level, units in (("graph", graph_units), ("configuration_mean", config_units)):
            target = np.asarray([unit["downstream_gap"] for unit in units], dtype=np.float64)
            for key, absolute in (("a", False), ("b", False), ("a", True)):
                x = np.asarray([unit[key] for unit in units], dtype=np.float64)
                y = target.copy()
                if absolute:
                    x, y = np.abs(x), np.abs(y)
                valid = np.isfinite(x) & np.isfinite(y)
                fig, ax = plt.subplots(figsize=(5.4, 4.4))
                for configuration in sorted({str(unit["configuration_id"]) for unit in units}):
                    selected = valid & np.array([str(unit["configuration_id"]) == configuration for unit in units])
                    ax.scatter(x[selected], y[selected], label=configuration, s=32, alpha=0.85)
                if valid.any():
                    lower = min(float(x[valid].min()), float(y[valid].min()))
                    upper = max(float(x[valid].max()), float(y[valid].max()))
                    padding = max((upper - lower) * 0.08, 0.001)
                    ax.plot([lower - padding, upper + padding], [lower - padding, upper + padding],
                            linestyle="--", color="0.6", linewidth=0.8)
                    if len({str(unit["configuration_id"]) for unit in units}) <= 10:
                        ax.legend(fontsize=7)
                else:
                    ax.text(0.5, 0.5, "No valid pair of gap estimates", ha="center", va="center", transform=ax.transAxes)
                target_name = "c" if label["metric"] == "sp" else "d"
                ax.set_xlabel(f"{'abs(' + key + ')' if absolute else key}: surrogate gap (score units)")
                ax.set_ylabel(f"{'abs(' + target_name + ')' if absolute else target_name}: final GNN gap (score units)")
                ax.set_title(f"{label['dataset']} / {label['metric'].upper()} / {label['phase']} / progress={label['progress']}\n"
                             f"{level}; valid={int(valid.sum())}/{len(units)}", fontsize=10)
                ax.grid(alpha=0.15)
                fig.text(0.02, 0.01, "Empirical alignment does not establish causal downstream descent.", fontsize=7)
                fig.tight_layout(rect=(0, 0.04, 1, 1))
                comparison = f"{'abs_' if absolute else 'signed_'}{key}_vs_{target_name}"
                stem = f"{slug}_{suffix}_{level}_{comparison}"
                for extension in ("png", "pdf"):
                    path = output_dir / f"{stem}.{extension}"
                    if path.exists():
                        raise FileExistsError(path)
                    fig.savefig(path, dpi=160)
                    paths.append(str(path))
                plt.close(fig)
    return paths


def write_analysis(records, output_dir, *, sign_threshold=SIGN_THRESHOLD, plots=True):
    """Write a new result directory, refusing to overwrite previous outputs."""
    records = list(records)
    summaries = summarize_records(records, sign_threshold=sign_threshold)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("gap_records.csv", "gap_summary.csv", "analysis_metadata.json"):
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)
    _write_csv(output_dir / "gap_records.csv", records)
    _write_csv(output_dir / "gap_summary.csv", summaries)
    plot_paths = _plot_groups(records, output_dir) if plots else []
    metadata = {"sign_threshold": sign_threshold, "score_unit": "probability", "pp_multiplier": 100,
                "full_cache_gnn_evaluation": False, "record_count": len(records),
                "summary_count": len(summaries), "plots": plot_paths,
                "graph_unit": "mean across evaluator seeds within one generated graph",
                "configuration_unit": "mean across graph means within controller seed, then across controller seeds",
                "invalid_repeat_policy": "nonfinite repeats invalidate the aggregation unit",
                "interpretation": "Positive correlation is empirical alignment under observed conditions; "
                                  "it does not guarantee causal downstream descent of the sampler."}
    with (output_dir / "analysis_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return {"records": records, "summary": summaries, "plot_paths": plot_paths,
            "gap_records": str(output_dir / "gap_records.csv"),
            "gap_summary": str(output_dir / "gap_summary.csv")}
