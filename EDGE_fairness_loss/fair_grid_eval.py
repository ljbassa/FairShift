#!/usr/bin/env python3
import argparse
import ast
import csv
import itertools
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


NON_METRIC_KEYS = {
    'eta', 'k', 'seed', 'returncode', 'parse_error', 'subprocess_error'
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Run multi-seed eta×k fairness grid for EDGE evaluate.py, aggregate results, '
            'and draw a Pareto curve.'
        )
    )
    p.add_argument('--repo_dir', type=str, required=True, help='Path to EDGE_fairness repo root')
    p.add_argument('--python_exec', type=str, required=True, help='Python executable for evaluate.py')
    p.add_argument('--eta_values', type=float, nargs='+', required=True)
    p.add_argument('--k_values', type=float, nargs='+', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--out_dir', type=str, default='fair_grid_results_multiseed')
    p.add_argument('--pair_mode', action='store_true', help='Use zip(eta_values, k_values) instead of full Cartesian product')
    p.add_argument('--include_baseline', action='store_true', help='Also add one extra baseline point with eta=0 and baseline_k')
    p.add_argument('--baseline_k', type=float, default=1.0, help='k value used for the extra baseline point when --include_baseline is set')
    p.add_argument('--fail_fast', action='store_true', help='Stop immediately if one subprocess call fails')
    p.add_argument('--plot_x_metric', type=str, default='value/fair_edge_sp_abs_gap', help='Metric to minimize on x-axis')
    p.add_argument('--plot_y_metric', type=str, default='value/linkpred_auc', help='Metric to maximize on y-axis')
    p.add_argument('--label_points', choices=['none', 'front', 'all'], default='front')
    p.add_argument('--plot_title', type=str, default='Pareto curve: fairness gap vs AUC')
    p.add_argument(
        'eval_args', nargs=argparse.REMAINDER,
        help=(
            'Arguments forwarded to evaluate.py. Put them after "--". '
            'Do NOT include --fair_score_eta, --fair_score_k, or --seed; the script injects them.'
        )
    )
    args = p.parse_args()
    if args.eval_args and args.eval_args[0] == '--':
        args.eval_args = args.eval_args[1:]
    return args


def make_combos(etas: Sequence[float], ks: Sequence[float], pair_mode: bool) -> List[Tuple[float, float]]:
    if pair_mode:
        if len(etas) != len(ks):
            raise ValueError('--pair_mode requires len(eta_values) == len(k_values)')
        return list(zip(etas, ks))
    return list(itertools.product(etas, ks))


def sanitize_float_tag(x: float) -> str:
    s = f'{x}'
    return s.replace('-', 'm').replace('.', 'p')


def strip_flag_with_value(argv: List[str], flag: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(x == flag for x in argv)


def prepare_base_eval_args(argv: List[str]) -> List[str]:
    if not argv:
        raise ValueError('No evaluate.py args were provided. Pass your known-good evaluate.py flags after "--".')

    cleaned = list(argv)
    if cleaned[0].endswith('evaluate.py'):
        cleaned = cleaned[1:]

    for flag in ('--fair_score_eta', '--fair_score_k', '--seed'):
        cleaned = strip_flag_with_value(cleaned, flag)

    if not has_flag(cleaned, '--fair_score_sp'):
        cleaned.append('--fair_score_sp')

    return cleaned


def extract_flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    prefix = f'{flag}='
    for i, value in enumerate(argv):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def format_plot_title(title: str, dataset: Optional[str]) -> str:
    if not dataset:
        return title
    if '{dataset}' in title:
        return title.format(dataset=dataset)
    if dataset.lower() in title.lower():
        return title
    return f'{dataset}: {title}'


def pareto_filename(dataset: Optional[str]) -> str:
    if not dataset:
        return 'pareto_curve.jpg'
    dataset_tag = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in dataset.strip())
    return f'pareto_curve_{dataset_tag}.jpg' if dataset_tag else 'pareto_curve.jpg'


def parse_metrics(stdout: str) -> Dict[str, float]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith('{') and line.endswith('}'):
            try:
                return ast.literal_eval(line)
            except Exception:
                return eval(line, {'__builtins__': {}}, {'nan': float('nan')})  # noqa: S307
    raise RuntimeError('Could not parse metrics dict from evaluate.py stdout')


def fmt_num(x):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return 'nan'
        if abs(x) >= 1e4 or (0 < abs(x) < 1e-4):
            return f'{x:.6e}'
        return f'{x:.6f}'
    return str(x)


def ordered_fieldnames(rows: List[Dict]) -> List[str]:
    preferred = [
        'eta', 'k', 'seed', 'value/linkpred_auc', 'nmae/linkpred_auc',
        'value/fair_edge_sp_abs_gap', 'value/fair_edge_sp_gap',
        'value/fair_edge_sensitive_rate', 'value/fair_edge_nonsensitive_rate',
        'value/fair_edge_score_sp_abs_gap', 'value/fair_edge_score_sp_gap',
        'value/fair_edge_score_sensitive_rate', 'value/fair_edge_score_nonsensitive_rate',
        'ref/fair_edge_sp_abs_gap', 'ref/fair_edge_sp_gap',
        'ref/fair_edge_score_sp_abs_gap', 'ref/fair_edge_score_sp_gap',
        'value/d', 'nmae/d', 'value/triangle_count', 'nmae/triangle_count',
        'value/clustering_coefficient', 'nmae/clustering_coefficient',
        'returncode', 'parse_error', 'subprocess_error'
    ]
    out: List[str] = []
    seen = set()
    for key in preferred:
        if any(key in row for row in rows):
            out.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                out.append(key)
                seen.add(key)
    return out


def write_csv(rows: List[Dict], path: Path):
    keys = ordered_fieldnames(rows)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(float(x))


def numeric_metric_keys(rows: List[Dict]) -> List[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if key in NON_METRIC_KEYS:
                continue
            if is_number(value):
                keys.add(key)
    return sorted(keys)


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[float, float], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row['eta'], row['k']), []).append(row)

    metric_keys = numeric_metric_keys([r for r in rows if r.get('returncode') == 0])
    agg_rows: List[Dict] = []
    for (eta, k), bucket in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        success_rows = [r for r in bucket if r.get('returncode') == 0]
        agg = {
            'eta': eta,
            'k': k,
            'n_runs': len(bucket),
            'n_success': len(success_rows),
            'seeds': ','.join(str(r['seed']) for r in sorted(bucket, key=lambda r: r['seed'])),
        }
        if not success_rows:
            agg_rows.append(agg)
            continue
        for key in metric_keys:
            vals = [float(r[key]) for r in success_rows if key in r and is_number(r[key])]
            if not vals:
                continue
            agg[f'{key}_mean'] = statistics.fmean(vals)
            agg[f'{key}_std'] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
    return agg_rows


def ordered_fieldnames_agg(rows: List[Dict]) -> List[str]:
    preferred = [
        'eta', 'k', 'n_runs', 'n_success', 'seeds',
        'value/linkpred_auc_mean', 'value/linkpred_auc_std',
        'value/fair_edge_sp_abs_gap_mean', 'value/fair_edge_sp_abs_gap_std',
        'value/fair_edge_sp_gap_mean', 'value/fair_edge_sp_gap_std',
        'value/fair_edge_sensitive_rate_mean', 'value/fair_edge_sensitive_rate_std',
        'value/fair_edge_nonsensitive_rate_mean', 'value/fair_edge_nonsensitive_rate_std',
        'value/fair_edge_score_sp_abs_gap_mean', 'value/fair_edge_score_sp_abs_gap_std',
        'value/fair_edge_score_sp_gap_mean', 'value/fair_edge_score_sp_gap_std',
        'value/fair_edge_score_sensitive_rate_mean', 'value/fair_edge_score_sensitive_rate_std',
        'value/fair_edge_score_nonsensitive_rate_mean', 'value/fair_edge_score_nonsensitive_rate_std',
        'value/d_mean', 'value/d_std',
        'value/triangle_count_mean', 'value/triangle_count_std',
        'value/clustering_coefficient_mean', 'value/clustering_coefficient_std',
        'nmae/linkpred_auc_mean', 'nmae/linkpred_auc_std',
        'nmae/d_mean', 'nmae/d_std',
        'nmae/triangle_count_mean', 'nmae/triangle_count_std',
        'nmae/clustering_coefficient_mean', 'nmae/clustering_coefficient_std',
        'ref/fair_edge_sp_abs_gap_mean', 'ref/fair_edge_sp_abs_gap_std',
        'ref/fair_edge_score_sp_gap_mean', 'ref/fair_edge_score_sp_gap_std',
        'ref/fair_edge_score_sp_abs_gap_mean', 'ref/fair_edge_score_sp_abs_gap_std',
    ]
    out: List[str] = []
    seen = set()
    for key in preferred:
        if any(key in row for row in rows):
            out.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                out.append(key)
                seen.add(key)
    return out


def write_csv_agg(rows: List[Dict], path: Path):
    keys = ordered_fieldnames_agg(rows)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pareto_front_indices(xs: Sequence[float], ys: Sequence[float]) -> List[int]:
    idxs = list(range(len(xs)))
    front: List[int] = []
    for i in idxs:
        dominated = False
        for j in idxs:
            if i == j:
                continue
            no_worse_x = xs[j] <= xs[i]
            no_worse_y = ys[j] >= ys[i]
            strictly_better = xs[j] < xs[i] or ys[j] > ys[i]
            if no_worse_x and no_worse_y and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    front.sort(key=lambda i: (xs[i], -ys[i]))
    return front


def plot_pareto(
    agg_rows: List[Dict],
    out_dir: Path,
    x_metric: str,
    y_metric: str,
    title: str,
    label_points: str,
    dataset: Optional[str] = None,
):
    x_mean_key = f'{x_metric}_mean'
    x_std_key = f'{x_metric}_std'
    y_mean_key = f'{y_metric}_mean'
    y_std_key = f'{y_metric}_std'

    valid = [
        row for row in agg_rows
        if is_number(row.get(x_mean_key)) and is_number(row.get(y_mean_key)) and row.get('n_success', 0) > 0
    ]
    if not valid:
        raise RuntimeError(
            f'No aggregated rows contain {x_mean_key} and {y_mean_key}. '
            'Check that fairness metrics are being emitted by evaluate.py.'
        )

    xs = [float(row[x_mean_key]) for row in valid]
    ys = [float(row[y_mean_key]) for row in valid]
    xerrs = [float(row.get(x_std_key, 0.0) or 0.0) for row in valid]
    yerrs = [float(row.get(y_std_key, 0.0) or 0.0) for row in valid]
    labels = [f"eta={fmt_num(row['eta'])}, k={fmt_num(row['k'])}" for row in valid]

    front_idx = pareto_front_indices(xs, ys)
    front_rows = [valid[i] for i in front_idx]

    plt.figure(figsize=(8, 6))
    plt.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt='o', capsize=3)

    front_x = [xs[i] for i in front_idx]
    front_y = [ys[i] for i in front_idx]
    plt.plot(front_x, front_y, linestyle='-', marker='o')

    if label_points in {'front', 'all'}:
        annotate_idx: Iterable[int]
        annotate_idx = range(len(valid)) if label_points == 'all' else front_idx
        for i in annotate_idx:
            plt.annotate(labels[i], (xs[i], ys[i]), xytext=(4, 4), textcoords='offset points', fontsize=8)

    plt.xlabel(f'{x_metric} (lower is better)')
    plt.ylabel(f'{y_metric} (higher is better)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = out_dir / pareto_filename(dataset)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def main():
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f'repo_dir does not exist: {repo_dir}')
    if not (repo_dir / 'evaluate.py').exists():
        raise FileNotFoundError(f'evaluate.py not found in repo_dir: {repo_dir}')

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_eval_args = prepare_base_eval_args(args.eval_args)
    combos = make_combos(args.eta_values, args.k_values, args.pair_mode)
    if args.include_baseline:
        combos = list({(float(eta), float(k)) for eta, k in combos} | {(0.0, float(args.baseline_k))})
        combos.sort(key=lambda x: (x[0], x[1]))
    rows: List[Dict] = []

    total_runs = len(combos) * len(args.seeds)
    print('Base evaluate.py args:')
    print('  ' + ' '.join(base_eval_args))
    print(f'Python exec: {args.python_exec}')
    print(f'Combos     : {len(combos)}')
    print(f'Seeds      : {args.seeds}')
    print(f'Total runs : {total_runs}')

    run_idx = 0
    for eta, k in combos:
        for seed in args.seeds:
            run_idx += 1
            cmd = [
                args.python_exec,
                'evaluate.py',
                *base_eval_args,
                '--seed', str(seed),
                '--fair_score_eta', str(eta),
                '--fair_score_k', str(k),
            ]

            print(f'[{run_idx}/{total_runs}] seed={seed}, eta={eta}, k={k}')
            proc = subprocess.run(
                cmd,
                cwd=str(repo_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            row = {
                'eta': eta,
                'k': k,
                'seed': seed,
                'returncode': proc.returncode,
            }

            if proc.returncode == 0:
                try:
                    row.update(parse_metrics(proc.stdout))
                except Exception as e:
                    row['parse_error'] = str(e)
            else:
                stderr_lines = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
                row['subprocess_error'] = stderr_lines[-1] if stderr_lines else 'evaluate.py failed'
                if args.fail_fast:
                    rows.append(row)
                    break
            rows.append(row)
        else:
            continue
        break

    rows_sorted = sorted(rows, key=lambda r: (r.get('eta', float('inf')), r.get('k', float('inf')), r.get('seed', float('inf'))))
    agg_rows = aggregate_rows(rows_sorted)
    dataset = extract_flag_value(base_eval_args, '--dataset')

    write_csv(rows_sorted, out_dir / 'summary_long.csv')

    plot_path = plot_pareto(
        agg_rows=agg_rows,
        out_dir=out_dir,
        x_metric=args.plot_x_metric,
        y_metric=args.plot_y_metric,
        title=format_plot_title(args.plot_title, dataset),
        label_points=args.label_points,
        dataset=dataset,
    )

    print('\nDone.')
    print(f'long csv   : {out_dir / "summary_long.csv"}')
    print(f'pareto jpg : {plot_path}')


if __name__ == '__main__':
    main()
