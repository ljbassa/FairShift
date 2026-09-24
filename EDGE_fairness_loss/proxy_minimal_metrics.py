"""CPU E1 diagnostics on the production cache and one terminal GCN per graph.

``decode_pairs`` accepts a CPU long tensor of shape [2, K] and returns K
terminal probabilities.  Full-cache predictions are reduced chunk by chunk;
there is no dense N x N decode and no intermediate-graph model fitting here.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _array(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _vector(value, size, name, dtype=np.float64):
    result = _array(value, dtype).reshape(-1)
    if result.size != size:
        raise ValueError(f"{name} has {result.size} entries; expected {size}")
    return result


def _pairs(value, num_nodes, name):
    result = _array(value, np.int64)
    if result.ndim != 2 or result.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2, E]")
    if result.size and ((result < 0).any() or (result >= num_nodes).any()):
        raise ValueError(f"{name} contains an out-of-range node ID")
    if (result[0] == result[1]).any():
        raise ValueError(f"{name} contains self pairs")
    return result


def _codes(pairs, num_nodes):
    return np.minimum(pairs[0], pairs[1]) * num_nodes + np.maximum(pairs[0], pairs[1])


def _membership(codes, sorted_support):
    if not len(sorted_support):
        return np.zeros(len(codes), dtype=bool)
    index = np.searchsorted(sorted_support, codes)
    return (index < len(sorted_support)) & (sorted_support[np.minimum(index, len(sorted_support) - 1)] == codes)


def _unique_codes(pairs, num_nodes, name):
    codes = _codes(pairs, num_nodes)
    sorted_codes = np.sort(codes)
    if len(codes) > 1 and (sorted_codes[1:] == sorted_codes[:-1]).any():
        raise ValueError(f"{name} contains duplicate undirected pairs")
    return codes, sorted_codes


def _test_labels(labels, test_pairs, positive_codes, num_nodes):
    y = _vector(labels, test_pairs.shape[1], "test_labels")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("test_labels must be binary")
    expected = _membership(_codes(test_pairs, num_nodes), positive_codes)
    if not np.array_equal(y.astype(bool), expected):
        raise ValueError("Test labels must describe the completed generated graph")
    return y


@dataclass
class _Gap:
    """Same-minus-different weighted means, preserving invalid observations."""

    min_mass: float
    mass: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    weighted_sum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    count: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.int64))
    finite_count: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.int64))
    invalid: bool = False

    def add(self, score, weight, same):
        score, weight, same = _array(score), _array(weight), _array(same, bool)
        if not np.isfinite(weight).all() or (weight < 0).any():
            self.invalid = True
        for group, mask in enumerate((same, ~same)):
            selected = mask & (weight > 0)
            finite = selected & np.isfinite(score) & np.isfinite(weight)
            self.count[group] += np.count_nonzero(selected)
            self.finite_count[group] += np.count_nonzero(finite)
            self.mass[group] += weight[mask].sum(dtype=np.float64)
            self.weighted_sum[group] += (score[finite] * weight[finite]).sum(dtype=np.float64)
            if np.count_nonzero(selected) != np.count_nonzero(finite):
                self.invalid = True

    def result(self, prefix):
        means = np.full(2, np.nan)
        if not self.invalid:
            np.divide(self.weighted_sum, self.mass, out=means, where=self.mass > self.min_mass)
        valid = bool(np.isfinite(means).all())
        result = {prefix: float(means[0] - means[1]) if valid else float("nan"),
                  f"{prefix}_valid_count": int(valid)}
        for index, group in enumerate(("same", "different")):
            result.update({f"{prefix}_{group}_mean": float(means[index]),
                           f"{prefix}_{group}_mass": float(self.mass[index]),
                           f"{prefix}_{group}_count": int(self.count[index]),
                           f"{prefix}_{group}_finite_count": int(self.finite_count[index])})
        return result


def safe_correlation(x, y, *, method="spearman"):
    """Finite paired count and NaN for missing/constant correlation inputs."""
    x, y = _array(x, np.float64).reshape(-1), _array(y, np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Correlation inputs have different sizes")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    result = {"value": float("nan"), "pair_count": int(len(x)),
              "total_pair_count": int(len(finite)), "valid_count": 0}
    if method not in ("spearman", "pearson"):
        raise ValueError("Unknown correlation method")
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return result
    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    result.update(value=float(np.corrcoef(x, y)[0, 1]), valid_count=1)
    return result


def decode_chunked(decode_pairs: Callable, pairs, *, chunk_size=65536):
    """Decode a small requested support (e.g. test pairs) with bounded batches."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    pairs = _array(pairs, np.int64)
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError("pairs must have shape [2, E]")
    result = np.empty(pairs.shape[1], dtype=np.float64)
    for start in range(0, pairs.shape[1], chunk_size):
        stop = min(start + chunk_size, pairs.shape[1])
        result[start:stop] = _vector(decode_pairs(torch.from_numpy(pairs[:, start:stop].copy())),
                                    stop - start, "decoded probabilities")
    return result


def terminal_metrics(test_scores, test_labels, test_pairs, node_groups,
                     generated_positive_pairs, num_nodes, *,
                     generated_positive_scores=None, min_mass=1e-6):
    """Generated-test AUC, SP, soft EO and relation/density diagnostics.

    EO uses generated held-out positives.  Group means are calculated within
    each graph before any graph-level aggregation.  A missing group is NaN.
    """
    test_pairs = _pairs(test_pairs, num_nodes, "test_pairs")
    positive_pairs = _pairs(generated_positive_pairs, num_nodes, "generated_positive_pairs")
    _, positive_codes = _unique_codes(positive_pairs, num_nodes, "generated_positive_pairs")
    _unique_codes(test_pairs, num_nodes, "test_pairs")
    y = _test_labels(test_labels, test_pairs, positive_codes, num_nodes)
    scores = _vector(test_scores, len(y), "test_scores")
    groups = _vector(node_groups, num_nodes, "node_groups", dtype=None)
    same = groups[test_pairs[0]] == groups[test_pairs[1]]
    positive_same = groups[positive_pairs[0]] == groups[positive_pairs[1]]
    total_possible = num_nodes * (num_nodes - 1) // 2
    result = {"num_nodes": int(num_nodes), "generated_positive_count": int(positive_pairs.shape[1]),
              "density": float(positive_pairs.shape[1] / total_possible) if total_possible else float("nan"),
              "density_valid_count": int(total_possible > 0),
              "test_pair_count": int(len(y)), "test_positive_count": int(y.sum()),
              "test_score_finite_count": int(np.isfinite(scores).sum())}
    auc_valid = bool(len(y) and np.isfinite(scores).all() and len(np.unique(y)) == 2)
    result.update(auc=float(roc_auc_score(y, scores)) if auc_valid else float("nan"),
                  auc_valid_count=int(auc_valid))
    for name, weights in (("sp", np.ones(len(y))), ("eo", y)):
        gap = _Gap(min_mass)
        gap.add(scores, weights, same)
        result.update(gap.result(name))
        result[f"{name}_abs"] = abs(result[name])
    for relation, mask in (("same", same), ("different", ~same)):
        selected = mask & y.astype(bool)
        result[f"test_positive_{relation}_count"] = int(selected.sum())
        result[f"test_positive_{relation}_score_mean"] = result[f"eo_{relation}_mean"]
        result[f"test_positive_{relation}_score_valid_count"] = int(selected.sum()) if np.isfinite(result[f"eo_{relation}_mean"]) else 0
    positive_scores = (np.full(positive_pairs.shape[1], np.nan) if generated_positive_scores is None
                       else _vector(generated_positive_scores, positive_pairs.shape[1], "generated_positive_scores"))
    for relation, mask in (("same", positive_same), ("different", ~positive_same)):
        finite = np.isfinite(positive_scores[mask])
        valid = bool(mask.any() and finite.all())
        result[f"generated_positive_{relation}_count"] = int(mask.sum())
        result[f"generated_positive_{relation}_score_mean"] = float(positive_scores[mask].mean()) if valid else float("nan")
        result[f"generated_positive_{relation}_score_valid_count"] = int(finite.sum())
    return result


def analyze_snapshot(snapshot: Mapping, *, decode_pairs: Callable,
                     generated_positive_pairs, test_pairs, test_labels,
                     num_nodes: int, chunk_size=65536, min_mass=1e-6):
    """Compute d/R/C/S from full production Q and terminal generated-test P.

    SP has v=y=1; EO uses the observed detached unshifted w for v and the
    completed generated graph's adjacency B for y.  Only correlation uses P
    for SP, and only generated test positives for EO.  Invalid summands stay
    NaN; the telescoping identity is never repaired by dropping bad groups.
    """
    if chunk_size <= 0 or min_mass < 0:
        raise ValueError("chunk_size must be positive and min_mass nonnegative")
    target = str(snapshot.get("target", snapshot.get("metric", ""))).lower()
    if target not in ("sp", "eo"):
        raise ValueError("snapshot metric must be sp or eo")
    pairs = _pairs(snapshot["pair_ids"], num_nodes, "pair_ids")
    size = pairs.shape[1]
    q = _vector(snapshot["q"], size, "q")
    same = _vector(snapshot["same_mask"], size, "same_mask", bool)
    w = _vector(snapshot["w"], size, "w") if target == "eo" else None
    codes, _ = _unique_codes(pairs, num_nodes, "production pair_ids")
    order = np.argsort(codes)
    sorted_codes = codes[order]
    positive_pairs = _pairs(generated_positive_pairs, num_nodes, "generated_positive_pairs")
    _, positive_codes = _unique_codes(positive_pairs, num_nodes, "generated_positive_pairs")
    if not _membership(positive_codes, sorted_codes).all():
        raise ValueError("Generated positive pair is absent from production cache support")
    test_pairs = _pairs(test_pairs, num_nodes, "test_pairs")
    test_codes, _ = _unique_codes(test_pairs, num_nodes, "test_pairs")
    if not _membership(test_codes, sorted_codes).all():
        raise ValueError("Generated test pair is absent from production cache support")
    y_p = _test_labels(test_labels, test_pairs, positive_codes, num_nodes)
    p_index = order[np.searchsorted(sorted_codes, test_codes)]
    q_p, same_p = q[p_index], same[p_index]
    accumulators = {name: _Gap(min_mass) for name in ("d", "R", "Q_v_g", "Q_y_g")}
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        g = _vector(decode_pairs(torch.from_numpy(pairs[:, start:stop].copy())), stop - start, "decoded probabilities")
        q_chunk, same_chunk = q[start:stop], same[start:stop]
        v = np.ones(stop - start) if target == "sp" else w[start:stop]
        y = np.ones(stop - start) if target == "sp" else _membership(codes[start:stop], positive_codes).astype(float)
        accumulators["d"].add(q_chunk, v, same_chunk)
        accumulators["R"].add(g - q_chunk, v, same_chunk)
        accumulators["Q_v_g"].add(g, v, same_chunk)
        accumulators["Q_y_g"].add(g, y, same_chunk)
    result: Dict = {"target": target, "full_pair_count": int(size),
                    "full_same_count": int(same.sum()), "full_different_count": int((~same).sum())}
    for key in ("requested_progress", "loop_index", "step_number", "diffusion_t", "total_steps", "progress", "guidance_applied"):
        if key in snapshot:
            value = snapshot[key]
            result[key] = _array(value).item() if isinstance(value, (np.ndarray, torch.Tensor)) else value
    for name, accumulator in accumulators.items():
        result.update(accumulator.result(name))
    g_p = decode_chunked(decode_pairs, test_pairs, chunk_size=chunk_size)
    p_gap = _Gap(min_mass)
    p_gap.add(g_p, np.ones(len(y_p)) if target == "sp" else y_p, same_p)
    result.update(p_gap.result("terminal_gap"))
    result["C"] = 0.0 if target == "sp" else result["Q_y_g"] - result["Q_v_g"]
    result["S"] = result["terminal_gap"] - result["Q_y_g"]
    for name in ("R", "C", "S"):
        result[f"{name}_abs"] = abs(result[name])
        result[f"{name}_valid_count"] = int(np.isfinite(result[name]))
    result["reconstructed_terminal_gap"] = sum(result[name] for name in ("d", "R", "C", "S"))
    result["identity_residual"] = result["terminal_gap"] - result["reconstructed_terminal_gap"]
    result["identity_valid_count"] = int(np.isfinite(result["identity_residual"]))
    result["identity_holds"] = bool(np.isfinite(result["identity_residual"]) and abs(result["identity_residual"]) <= 1e-8)
    signed_valid = bool(np.isfinite(result["d"]) and np.isfinite(result["terminal_gap"]))
    result["signed_agreement"] = float(np.sign(result["d"]) == np.sign(result["terminal_gap"])) if signed_valid else float("nan")
    result["signed_agreement_valid_count"] = int(signed_valid)
    selected = np.ones(len(y_p), dtype=bool) if target == "sp" else y_p.astype(bool)
    correlation = safe_correlation(q_p[selected], g_p[selected])
    result.update(q_g_spearman=correlation["value"], q_g_pair_count=correlation["pair_count"],
                  q_g_total_pair_count=correlation["total_pair_count"], q_g_valid_count=correlation["valid_count"])
    if "production_proxy" in snapshot:
        proxy = _array(snapshot["production_proxy"], np.float64).reshape(-1)
        if len(proxy) != 1:
            raise ValueError("E1 analyzes one completed graph per observer snapshot")
        production_valid = bool(_array(snapshot.get("valid_graph", [True]), bool).reshape(-1)[0])
        result["production_proxy"] = float(proxy[0]) if production_valid else float("nan")
        result["production_proxy_valid_count"] = int(production_valid and np.isfinite(proxy[0]))
        result["production_proxy_residual"] = result["d"] - result["production_proxy"]
    return result


def _aggregate(rows, metrics):
    result = {"graph_count": len(rows)}
    for metric in metrics:
        values = np.asarray([row.get(metric, float("nan")) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[f"{metric}_valid_count"] = len(finite)
        result[f"{metric}_mean"] = float(finite.mean()) if len(finite) else float("nan")
        result[f"{metric}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
        result[f"{metric}_sem"] = float(finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else float("nan")
    return result


def summarize_rows(terminal_rows: Sequence[Mapping], alignment_rows: Sequence[Mapping]):
    """Independent graph/root summaries; never pool targets or timesteps.

    Each input row must include root_seed.  Terminal rows use arm (falling
    back to target); alignment rows use target and requested_progress.
    Repeated root rows within one group are rejected rather than counted as
    independent observations.  Cross-root correlations report graph counts.
    """
    terminal_groups, alignment_groups = {}, {}
    for row in terminal_rows:
        key = row.get("arm", row.get("target"))
        if key is None:
            raise ValueError("terminal rows require arm or target")
        terminal_groups.setdefault(key, []).append(row)
    for row in alignment_rows:
        key = (row["target"], row["requested_progress"])
        if key[0] not in ("sp", "eo"):
            raise ValueError("alignment rows contain a non-guided target")
        alignment_groups.setdefault(key, []).append(row)
    output = {"terminal": [], "alignment": []}
    for group in (*terminal_groups.values(), *alignment_groups.values()):
        seeds = [row["root_seed"] for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate root_seed within an arm or target/timestep")
    terminal_columns = ("auc", "sp", "sp_abs", "eo", "eo_abs", "density",
                        "generated_positive_same_count", "generated_positive_different_count",
                        "generated_positive_same_score_mean", "generated_positive_different_score_mean",
                        "test_positive_same_count", "test_positive_different_count",
                        "test_positive_same_score_mean", "test_positive_different_score_mean")
    for arm, rows in sorted(terminal_groups.items()):
        output["terminal"].append({"arm": arm, **_aggregate(rows, terminal_columns)})
    for (target, progress), rows in sorted(alignment_groups.items()):
        aggregate = {"target": target, "requested_progress": progress,
                     **_aggregate(rows, ("d", "terminal_gap", "R_abs", "C_abs", "S_abs",
                                         "q_g_spearman", "signed_agreement", "identity_residual"))}
        for method in ("pearson", "spearman"):
            correlation = safe_correlation([row["d"] for row in rows],
                                           [row["terminal_gap"] for row in rows], method=method)
            prefix = f"signed_proxy_terminal_{method}"
            aggregate.update({prefix: correlation["value"],
                              f"{prefix}_graph_count": correlation["pair_count"],
                              f"{prefix}_valid_count": correlation["valid_count"]})
        output["alignment"].append(aggregate)
    return output
