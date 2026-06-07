#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_X_METRIC = "lp/sp_abs_gap_mean"
DEFAULT_Y_METRIC = "lp/auc"
DEFAULT_GRAPHS_DIR = "generated_graphs"


@dataclass
class PlotPoint:
    label: str
    source: Path
    x: float
    y: float
    xerr: Optional[float]
    yerr: Optional[float]
    x_metric: str
    y_metric: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find summary CSV files in generated_graphs and draw a scatter plot "
            "using LP fairness and AUC metrics."
        )
    )
    parser.add_argument(
        "csv_names",
        nargs="*",
        help=(
            "CSV names, stems, paths, or glob patterns. Examples: "
            "cora_01, cora_01.summary.csv, generated_graphs/cora_*.summary.csv, '*.summary.csv'"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every *.summary.csv file in the graphs directory.",
    )
    parser.add_argument(
        "--graphs_dir",
        type=str,
        default=DEFAULT_GRAPHS_DIR,
        help="Directory containing summary CSV files. Default: generated_graphs",
    )
    parser.add_argument(
        "--x_metric",
        type=str,
        default=DEFAULT_X_METRIC,
        help="Metric for the x-axis. Default: lp/sp_abs_gap_mean",
    )
    parser.add_argument(
        "--y_metric",
        type=str,
        default=DEFAULT_Y_METRIC,
        help="Metric for the y-axis. Default: lp/auc, resolved to lp/auc_mean when needed.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. Default: generated_graphs/lp_auc_vs_sp_abs_gap.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="LP AUC vs SP absolute gap",
        help="Plot title.",
    )
    parser.add_argument(
        "--no_error_bars",
        action="store_true",
        help="Do not draw *_std columns as error bars.",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Do not write CSV names next to points.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save a PDF next to the PNG.",
    )
    args = parser.parse_args()

    if not args.all and not args.csv_names:
        parser.error("provide at least one CSV name/pattern, or use --all")
    return args


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_graphs_dir(graphs_dir: str) -> Path:
    path = Path(graphs_dir).expanduser()
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path, repo_root() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def has_glob_chars(value: str) -> bool:
    return any(ch in value for ch in "*?[]")


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def candidate_paths(name: str, graphs_dir: Path) -> List[Path]:
    raw = Path(name).expanduser()
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    elif raw.parent != Path("."):
        candidates.extend([Path.cwd() / raw, repo_root() / raw])
    else:
        candidates.append(graphs_dir / raw)

    if raw.parent == Path("."):
        if not name.endswith(".csv"):
            candidates.append(graphs_dir / f"{name}.summary.csv")
            candidates.append(graphs_dir / f"{name}.csv")
        elif name.endswith(".summary"):
            candidates.append(graphs_dir / f"{name}.csv")

    return unique_paths(candidates)


def expand_input_name(name: str, graphs_dir: Path) -> List[Path]:
    if name.lower() in {"all", ":all"}:
        return sorted(path.resolve() for path in graphs_dir.glob("*.summary.csv") if path.is_file())

    if has_glob_chars(name):
        raw = Path(name).expanduser()
        matches: List[Path] = []
        if raw.is_absolute():
            matches.extend(Path(path).resolve() for path in raw.parent.glob(raw.name) if path.is_file())
        elif raw.parent != Path("."):
            matches.extend((Path.cwd() / raw.parent).glob(raw.name))
            matches.extend((repo_root() / raw.parent).glob(raw.name))
        else:
            matches.extend(graphs_dir.glob(name))
        return sorted(path.resolve() for path in unique_paths(matches) if path.is_file())

    return [path for path in candidate_paths(name, graphs_dir) if path.is_file()]


def find_csvs(csv_names: Sequence[str], graphs_dir: Path, include_all: bool) -> List[Path]:
    matches: List[Path] = []
    if include_all:
        matches.extend(path.resolve() for path in graphs_dir.glob("*.summary.csv") if path.is_file())

    missing = []
    for name in csv_names:
        found = expand_input_name(name, graphs_dir)
        if not found:
            missing.append(name)
        matches.extend(found)

    paths = unique_paths(matches)
    if missing:
        available = ", ".join(path.name for path in sorted(graphs_dir.glob("*.summary.csv")))
        raise FileNotFoundError(
            "Could not find CSV(s): "
            + ", ".join(missing)
            + f"\nSearched in: {graphs_dir}"
            + (f"\nAvailable summary CSVs: {available}" if available else "")
        )
    if not paths:
        raise FileNotFoundError(f"No summary CSV files found in: {graphs_dir}")
    return sorted(paths)


def parse_float(value: str, metric: str, source: Path) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: metric {metric!r} has non-numeric value {value!r}")
    if not math.isfinite(out):
        raise ValueError(f"{source}: metric {metric!r} is not finite: {value!r}")
    return out


def parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def resolve_metric(fieldnames: Sequence[str], requested: str) -> str:
    if requested in fieldnames:
        return requested

    candidates = []
    if not requested.endswith("_mean"):
        candidates.append(f"{requested}_mean")
    if requested == "lp/auc":
        candidates.append("lp/auc_mean")
    if requested == "lp/sp_abs_gap":
        candidates.append("lp/sp_abs_gap_mean")

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    preview = ", ".join(fieldnames[:12])
    if len(fieldnames) > 12:
        preview += ", ..."
    raise KeyError(f"Metric {requested!r} not found. Available columns include: {preview}")


def std_metric_for(fieldnames: Sequence[str], metric: str) -> Optional[str]:
    candidates = []
    if metric.endswith("_mean"):
        candidates.append(f"{metric[:-5]}_std")
    candidates.append(f"{metric}_std")

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def label_for(path: Path, row_index: int, row_count: int) -> str:
    name = path.name
    if name.endswith(".summary.csv"):
        name = name[: -len(".summary.csv")]
    elif name.endswith(".csv"):
        name = name[: -len(".csv")]

    if row_count <= 1:
        return name
    return f"{name}[{row_index}]"


def read_points(path: Path, x_metric: str, y_metric: str, use_error_bars: bool) -> List[PlotPoint]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"{path}: CSV has no header")

        x_key = resolve_metric(fieldnames, x_metric)
        y_key = resolve_metric(fieldnames, y_metric)
        xerr_key = std_metric_for(fieldnames, x_key) if use_error_bars else None
        yerr_key = std_metric_for(fieldnames, y_key) if use_error_bars else None
        rows: List[Dict[str, str]] = list(reader)

    if not rows:
        raise ValueError(f"{path}: CSV has no data rows")

    points = []
    for row_index, row in enumerate(rows):
        x = parse_float(row.get(x_key, ""), x_key, path)
        y = parse_float(row.get(y_key, ""), y_key, path)
        points.append(
            PlotPoint(
                label=label_for(path, row_index, len(rows)),
                source=path,
                x=x,
                y=y,
                xerr=parse_optional_float(row.get(xerr_key)) if xerr_key else None,
                yerr=parse_optional_float(row.get(yerr_key)) if yerr_key else None,
                x_metric=x_key,
                y_metric=y_key,
            )
        )
    return points


def default_out_path(graphs_dir: Path, csv_paths: Sequence[Path]) -> Path:
    if len(csv_paths) == 1:
        label = label_for(csv_paths[0], 0, 1)
        return graphs_dir / f"{label}.lp_auc_vs_sp_abs_gap.png"
    return graphs_dir / "lp_auc_vs_sp_abs_gap.png"


def axis_label(metric: str, better_text: str) -> str:
    return f"{metric} ({better_text})"


def plot_points(
    points: Sequence[PlotPoint],
    out_path: Path,
    title: str,
    show_error_bars: bool,
    show_labels: bool,
) -> None:
    if not points:
        raise ValueError("No points to plot")

    x_metrics = sorted(set(point.x_metric for point in points))
    y_metrics = sorted(set(point.y_metric for point in points))
    x_label = x_metrics[0] if len(x_metrics) == 1 else "x metric"
    y_label = y_metrics[0] if len(y_metrics) == 1 else "y metric"

    fig, ax = plt.subplots(figsize=(8, 6))
    for point in points:
        xerr = point.xerr if show_error_bars else None
        yerr = point.yerr if show_error_bars else None
        if xerr is not None or yerr is not None:
            ax.errorbar(
                point.x,
                point.y,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                capsize=3,
                alpha=0.85,
            )
        else:
            ax.scatter(point.x, point.y, alpha=0.85)

        if show_labels:
            ax.annotate(
                point.label,
                (point.x, point.y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel(axis_label(x_label, "lower is better"))
    ax.set_ylabel(axis_label(y_label, "higher is better"))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    graphs_dir = resolve_graphs_dir(args.graphs_dir)
    csv_paths = find_csvs(args.csv_names, graphs_dir, args.all)

    points: List[PlotPoint] = []
    for csv_path in csv_paths:
        points.extend(read_points(csv_path, args.x_metric, args.y_metric, not args.no_error_bars))

    out_path = Path(args.out).expanduser() if args.out else default_out_path(graphs_dir, csv_paths)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    plot_points(
        points=points,
        out_path=out_path,
        title=args.title,
        show_error_bars=not args.no_error_bars,
        show_labels=not args.no_labels,
    )
    print(f"Saved PNG: {out_path}")

    if args.pdf:
        pdf_path = out_path.with_suffix(".pdf")
        plot_points(
            points=points,
            out_path=pdf_path,
            title=args.title,
            show_error_bars=not args.no_error_bars,
            show_labels=not args.no_labels,
        )
        print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
