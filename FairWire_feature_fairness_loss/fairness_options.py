"""Metric selection and output isolation shared by controller entry points."""

from pathlib import Path


def metric_directory(path, metric):
    """Place results under the selected metric without duplicating its suffix."""
    path = Path(path)
    return path if path.name == metric else path / metric


def metric_file(path, metric):
    path = Path(path)
    return metric_directory(path.parent, metric) / path.name


def resolve_controller_metric(requested, saved=None):
    metric = saved or "sp"
    if requested is not None and saved is not None and requested != saved:
        raise ValueError(
            f"Controller checkpoint uses {saved!r}, but --fair_score_metric={requested!r}."
        )
    metric = requested or metric
    if metric not in {"sp", "eo"}:
        raise ValueError(f"Unknown fairness metric {metric!r}; expected 'sp' or 'eo'.")
    return metric
