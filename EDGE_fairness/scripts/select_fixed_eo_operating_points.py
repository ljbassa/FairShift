#!/usr/bin/env python3
"""Select EO operating points using matched generation seeds and an uncontrolled baseline.

Each seed's input metrics are means over generated graphs. Output standard
deviations are population standard deviations across those seed means, not
standard deviations across individual graphs.
"""

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys


FIELDS = [
    "dataset", "role", "status", "eta", "k", "seeds", "num_seeds",
    "num_evaluated_graphs_per_seed", "auc_seed_mean", "auc_seed_std",
    "eo_seed_mean", "eo_seed_std", "auc_drop_from_uncontrolled",
    "eo_improvement_from_uncontrolled", "auc_max_drop", "source_csv",
]


def finite_number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(key))
    return value


def integer(row, key):
    value = finite_number(row, key)
    if not value.is_integer():
        raise ValueError("{} must be an integer".format(key))
    return int(value)


def validated_metrics(row):
    if "fair_score_metric" in row and row["fair_score_metric"] != "eo":
        raise ValueError("fair_score_metric must be eo")
    if any(integer(row, key) != 0 for key in (
        "generate_returncode", "generated_eval_returncode"
    )):
        raise ValueError("generation or evaluation failed")
    count = integer(row, "num_evaluated_graphs")
    if count <= 0:
        raise ValueError("no evaluated graphs")
    if row.get("num_loaded_graphs") not in (None, ""):
        if integer(row, "num_loaded_graphs") != count:
            raise ValueError("some loaded graphs were not evaluated")
    if "aggregate_lp/num_graphs" in row:
        if integer(row, "aggregate_lp/num_graphs") != count:
            raise ValueError("LP evaluation did not succeed for every graph")
    auc = finite_number(row, "lp/auc_mean")
    eo = finite_number(row, "lp/eo_abs_gap_mean")
    for key in ("lp/auc_std", "lp/eo_abs_gap_std"):
        if key in row and finite_number(row, key) < 0:
            raise ValueError("negative standard deviation")
    for key in ("lp/eo_defined", "lp/eo_defined_mean", "eo_defined", "eo_defined_mean"):
        if key in row and finite_number(row, key) != 1:
            raise ValueError("EO is undefined for some graphs")
    return auc, eo, count


def select_operating_points(rows, dataset, source_csv, auc_max_drop=0.01):
    if not math.isfinite(auc_max_drop) or auc_max_drop < 0:
        raise ValueError("auc_max_drop must be finite and nonnegative")
    groups = {}
    all_seeds = set()
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        setting = (finite_number(row, "eta"), finite_number(row, "k"))
        seed = integer(row, "seed")
        group = groups.setdefault(setting, {})
        if seed in group:
            raise ValueError("duplicate eta/k/seed: {}/{}/{}".format(*setting, seed))
        group[seed] = row
        all_seeds.add(seed)
    baseline_key = (0.0, 1.0)
    if baseline_key not in groups or set(groups[baseline_key]) != all_seeds:
        raise ValueError("uncontrolled (eta=0, k=1) must cover the full seed set")
    seeds = sorted(all_seeds)

    def aggregate(setting, expected_count=None):
        group = groups[setting]
        if set(group) != all_seeds:
            raise ValueError("incomplete seed set")
        metrics = [validated_metrics(group[seed]) for seed in seeds]
        counts = {item[2] for item in metrics}
        if len(counts) != 1 or (expected_count is not None and counts != {expected_count}):
            raise ValueError("evaluated graph counts do not match uncontrolled")
        aucs, eos = [item[0] for item in metrics], [item[1] for item in metrics]
        return {
            "eta": setting[0], "k": setting[1], "seeds": ";".join(map(str, seeds)),
            "num_seeds": len(seeds), "num_evaluated_graphs_per_seed": metrics[0][2],
            "auc_seed_mean": statistics.mean(aucs), "auc_seed_std": statistics.pstdev(aucs),
            "eo_seed_mean": statistics.mean(eos), "eo_seed_std": statistics.pstdev(eos),
        }

    try:
        baseline = aggregate(baseline_key)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid uncontrolled baseline: {}".format(exc)) from exc
    candidates = []
    for setting in sorted(groups):
        if setting == baseline_key:
            continue
        try:
            candidate = aggregate(setting, baseline["num_evaluated_graphs_per_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            print("Skipping eta={}, k={}: {}".format(*setting, exc), file=sys.stderr)
            continue
        if candidate["eo_seed_mean"] < baseline["eo_seed_mean"]:
            candidates.append(candidate)
    candidates.sort(key=lambda row: (
        row["eo_seed_mean"], -row["auc_seed_mean"], row["eta"], row["k"]
    ))
    retained = [row for row in candidates if
                row["auc_seed_mean"] >= baseline["auc_seed_mean"] - auc_max_drop]
    output = []
    for role, selected in (
        ("uncontrolled", baseline), ("auc_retained", next(iter(retained), None)),
        ("fairness_oriented", next(iter(candidates), None)),
    ):
        result = dict.fromkeys(FIELDS, "")
        result.update(dataset=dataset, role=role, status="no_qualifying_run",
                      auc_max_drop=auc_max_drop, source_csv=str(Path(source_csv).resolve()))
        if selected is not None:
            result.update(selected)
            result.update(
                status="selected",
                auc_drop_from_uncontrolled=baseline["auc_seed_mean"] - selected["auc_seed_mean"],
                eo_improvement_from_uncontrolled=baseline["eo_seed_mean"] - selected["eo_seed_mean"],
            )
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--auc_max_drop", type=float, default=0.01)
    args = parser.parse_args()
    with args.summary_csv.open(newline="", encoding="utf-8") as handle:
        try:
            selected = select_operating_points(list(csv.DictReader(handle)), args.dataset,
                                               args.summary_csv, args.auc_max_drop)
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    print("Saved operating points: {}".format(args.out_csv))


if __name__ == "__main__":
    main()
