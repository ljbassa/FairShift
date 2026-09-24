#!/usr/bin/env python3
"""Select EO operating points against an independently generated uncontrolled baseline."""
import argparse
import csv
import math
import warnings
from pathlib import Path

AUC = "lp/auc_mean"
EO = "lp/eo_abs_gap_mean"
EO_DEFINED = "lp/eo_defined_mean"


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def valid_metrics(row):
    return (
        finite_number(row.get(AUC))
        and finite_number(row.get(EO))
        and (EO_DEFINED not in row or (
            finite_number(row[EO_DEFINED]) and float(row[EO_DEFINED]) == 1.0
        ))
    )


def select_points(rows, baseline, dataset, auc_max_drop=0.01,
                  fairness_max_auc_drop=None, summary_csv="", uncontrolled_summary_csv=""):
    for name, value in (("auc_max_drop", auc_max_drop),
                        ("fairness_max_auc_drop", fairness_max_auc_drop)):
        if value is not None and (not finite_number(value) or float(value) < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not valid_metrics(baseline):
        raise ValueError("Uncontrolled baseline requires finite AUC/EO and fully defined EO")
    baseline_auc, baseline_eo = float(baseline[AUC]), float(baseline[EO])
    eligible = [row for row in rows if valid_metrics(row)]
    if len(eligible) != len(rows):
        warnings.warn(
            f"Excluded {len(rows) - len(eligible)} rows with non-finite metrics or undefined EO",
            RuntimeWarning,
        )
    improved = [row for row in eligible if float(row[EO]) < baseline_eo]
    auc_floor = baseline_auc - auc_max_drop
    fairness_floor = None if fairness_max_auc_drop is None else baseline_auc - fairness_max_auc_drop
    rank = lambda row: (float(row[EO]), -float(row[AUC]), row.get("run_name", ""))
    pools = {
        "auc_retained": [row for row in improved if float(row[AUC]) >= auc_floor - 1e-12],
        "fairness_oriented": [row for row in improved if fairness_floor is None
                              or float(row[AUC]) >= fairness_floor - 1e-12],
    }
    provenance = {
        "dataset": dataset,
        "summary_csv": str(summary_csv),
        "uncontrolled_summary_csv": str(uncontrolled_summary_csv),
        "uncontrolled_auc": baseline_auc,
        "uncontrolled_eo": baseline_eo,
        "auc_max_drop": auc_max_drop,
        "auc_retained_floor": auc_floor,
        "fairness_max_auc_drop": "" if fairness_max_auc_drop is None else fairness_max_auc_drop,
        "fairness_auc_floor": "" if fairness_floor is None else fairness_floor,
    }
    selected = []
    choices = [("uncontrolled", dict(baseline, run_name=baseline.get("run_name") or "uncontrolled"))]
    choices += [(role, min(pool, key=rank) if pool else None) for role, pool in pools.items()]
    for role, source in choices:
        row = dict(source or {})
        row.update(provenance)
        row.update(selection=role, status="ok" if source is not None else "no_qualifying_run")
        if source is None:
            row.update(run_name="", **{AUC: "", EO: "", "lp/auc_std": "", "lp/eo_abs_gap_std": ""})
            warnings.warn(
                f"{dataset}: no qualifying {role} run improves uncontrolled EO within its AUC constraint",
                RuntimeWarning,
            )
        else:
            row[AUC], row[EO] = float(source[AUC]), float(source[EO])
            row["auc_drop_from_uncontrolled"] = baseline_auc - row[AUC]
            row["eo_reduction_from_uncontrolled"] = baseline_eo - row[EO]
            row["eo_reduction_from_uncontrolled_pct"] = (
                100.0 * (baseline_eo - row[EO]) / baseline_eo if baseline_eo else ""
            )
        selected.append(row)
    return selected


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = ["dataset", "selection", "status", "run_name", AUC, "lp/auc_std", EO, "lp/eo_abs_gap_std"]
    fields = list(dict.fromkeys(preferred + [key for row in rows for key in row]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_points(path, rows, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    valid = [row for row in rows if valid_metrics(row)]
    ax.scatter([float(row[EO]) for row in valid], [float(row[AUC]) for row in valid],
               s=24, color="#718096", alpha=0.45, label="EO grid")
    ax.axhline(selected[0]["auc_retained_floor"], color="#3182ce", linestyle="--",
               linewidth=1, label="AUC-retained floor (uncontrolled minus allowed drop)")
    for row, color, marker, size in zip(selected, ("#805ad5", "#3182ce", "#dd6b20"),
                                        ("*", "s", "D"), (200, 150, 75)):
        if row["status"] != "ok":
            continue
        ax.scatter(row[EO], row[AUC], s=size, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=1.8, label=row["selection"].replace("_", " "), zorder=5)
    ax.set(title=f"{selected[0]['dataset'].capitalize()}: AUC–EO operating points",
           xlabel="EO absolute score gap (lower is better)", ylabel="LP AUC (higher is better)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--uncontrolled_summary_csv", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--auc_max_drop", type=float, default=0.01,
                        help="Maximum absolute AUC drop from uncontrolled for AUC-retained (default: 0.01).")
    parser.add_argument("--fairness_max_auc_drop", type=float, default=None,
                        help="Optional maximum AUC drop from uncontrolled for fairness-oriented.")
    parser.add_argument("--plot_path", type=Path)
    args = parser.parse_args(argv)
    rows = read_rows(args.summary_csv)
    baseline_rows = read_rows(args.uncontrolled_summary_csv)
    if len(baseline_rows) != 1:
        parser.error("--uncontrolled_summary_csv must contain exactly one evaluated baseline row")
    for row in rows + baseline_rows:
        if row.get("dataset") and row["dataset"].lower() != args.dataset.lower():
            parser.error(f"Dataset mismatch: {row['dataset']!r} != {args.dataset!r}")
    if any(row.get("fair_score_metric") not in (None, "", "eo") for row in rows):
        parser.error("--summary_csv must contain EO controller runs")
    try:
        selected = select_points(
            rows, baseline_rows[0], args.dataset, args.auc_max_drop, args.fairness_max_auc_drop,
            args.summary_csv.resolve(), args.uncontrolled_summary_csv.resolve(),
        )
    except ValueError as exc:
        parser.error(str(exc))
    write_csv(args.out_csv, selected)
    if args.plot_path is not None:
        plot_points(args.plot_path, rows, selected)
    for row in selected:
        if row["status"] == "ok":
            print(f"{args.dataset}: {row['selection']}: AUC={row[AUC]:.6f}, EO={row[EO]:.6f}, {row['run_name']}")
        else:
            print(f"{args.dataset}: {row['selection']}: {row['status']}")
    print(f"Selected operating points: {args.out_csv}")


if __name__ == "__main__":
    main()
