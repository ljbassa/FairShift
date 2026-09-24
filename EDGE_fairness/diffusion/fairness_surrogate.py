"""Differentiable group-fairness proxies for reverse edge probabilities.

The EO proxy uses the frozen denoiser's unguided probability ``w_e`` as a soft
positive-edge condition and the controller's running probability ``q_e`` as
the score being equalized:

    E[q | Y=1, group=g] ~= sum_{e in g} w_e q_e / sum_{e in g} w_e.

``w_e`` is stop-gradient, so the controller cannot improve EO by moving the
conditioning set itself.  This avoids making a hard, non-differentiable
positive-edge mask during reverse diffusion.
"""

import torch


FAIR_SCORE_METRICS = ("sp", "eo")


def normalize_fair_score_metric(metric):
    metric = str(metric).strip().lower()
    if metric not in FAIR_SCORE_METRICS:
        raise ValueError(
            f"Unknown fair_score_metric={metric!r}; expected one of {FAIR_SCORE_METRICS}."
        )
    return metric


def group_fairness_terms(
    metric,
    score_same,
    score_diff,
    positive_mass_same,
    positive_mass_diff,
    positive_score_same,
    positive_score_diff,
    count_same,
    count_diff,
    *,
    q_active=None,
    positive_weight_active=None,
    batch_active=None,
    same_active=None,
    min_positive_mass=1e-6,
):
    """Return a signed fairness gap and, optionally, its derivative per edge.

    SP uses ``score / count``.  EO uses ``positive_score / positive_mass``,
    where positive_score is sum(w*q) and positive_mass is sum(w) from the
    unguided frozen-denoiser condition.

    A graph is invalid when either comparison group is empty.  EO additionally
    requires non-negligible soft-positive mass in both groups.  Invalid graphs
    receive exactly zero gap and zero derivative.  Unit denominators are used
    only on masked-out graphs for safe arithmetic, never as pseudo-count mass.
    """
    metric = normalize_fair_score_metric(metric)
    if score_same.shape != score_diff.shape:
        raise ValueError("score_same and score_diff must have identical shapes")

    count_valid = (count_same > 0) & (count_diff > 0)
    if metric == "sp":
        same_valid = count_same > 0
        diff_valid = count_diff > 0
        same_denom = torch.where(same_valid, count_same, torch.ones_like(count_same))
        diff_denom = torch.where(diff_valid, count_diff, torch.ones_like(count_diff))
        same_rate = score_same / same_denom
        diff_rate = score_diff / diff_denom
        valid_graph = count_valid
        support_same = count_same
        support_diff = count_diff
    else:
        threshold = torch.as_tensor(
            min_positive_mass,
            dtype=positive_mass_same.dtype,
            device=positive_mass_same.device,
        )
        same_valid = positive_mass_same > threshold
        diff_valid = positive_mass_diff > threshold
        same_denom = torch.where(
            same_valid, positive_mass_same, torch.ones_like(positive_mass_same)
        )
        diff_denom = torch.where(
            diff_valid, positive_mass_diff, torch.ones_like(positive_mass_diff)
        )
        same_rate = positive_score_same / same_denom
        diff_rate = positive_score_diff / diff_denom
        valid_graph = count_valid & same_valid & diff_valid
        support_same = positive_mass_same
        support_diff = positive_mass_diff

    raw_gap = same_rate - diff_rate
    gap = torch.where(valid_graph, raw_gap, torch.zeros_like(raw_gap))
    gap = torch.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)

    derivative = None
    if q_active is not None:
        if batch_active is None or same_active is None:
            raise ValueError("batch_active and same_active are required with q_active")
        same_float = same_active.to(dtype=q_active.dtype)
        diff_float = (~same_active).to(dtype=q_active.dtype)

        if metric == "sp":
            derivative = (
                same_float / same_denom.index_select(0, batch_active)
                - diff_float / diff_denom.index_select(0, batch_active)
            )
        else:
            if positive_weight_active is None:
                raise ValueError("positive_weight_active is required for EO derivatives")
            same_denom_active = same_denom.index_select(0, batch_active)
            diff_denom_active = diff_denom.index_select(0, batch_active)

            positive_weight_active = positive_weight_active.to(dtype=q_active.dtype)
            same_derivative = positive_weight_active / same_denom_active
            diff_derivative = positive_weight_active / diff_denom_active
            derivative = same_float * same_derivative - diff_float * diff_derivative

        derivative = torch.where(
            valid_graph.index_select(0, batch_active),
            derivative,
            torch.zeros_like(derivative),
        )
        derivative = torch.nan_to_num(derivative, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "gap": gap,
        "derivative": derivative,
        "valid_graph": valid_graph,
        "same_rate": same_rate,
        "diff_rate": diff_rate,
        "support_same": support_same,
        "support_diff": support_diff,
    }
