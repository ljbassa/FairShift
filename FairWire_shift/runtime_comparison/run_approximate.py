"""Run bounded timing-only preparation followed by unchanged FairShift pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--steps', type=int, default=32)
    args = parser.parse_args()
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu,
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'TF_CPP_MIN_LOG_LEVEL': '3'}
    destination = HERE / 'approximate'
    destination.mkdir(exist_ok=True)
    for dataset in args.datasets:
        checkpoint = destination / f'{dataset}_warmup{args.steps}.pth'
        prepare = [PYTHON, '-u', str(HERE / 'prepare_timing_checkpoint.py'),
                   '--dataset', dataset, '--output', str(checkpoint),
                   '--updates', str(args.steps)]
        pipeline = [PYTHON, '-u', str(HERE / 'benchmark_pipeline.py'),
                    '--checkpoint', str(checkpoint), '--dataset', dataset,
                    '--eta', '1', '--seed', '0', '--num-samples', '1',
                    '--output-dir', str(destination / dataset)]
        if checkpoint.exists() or (destination / dataset).exists():
            raise FileExistsError(f'Use a fresh destination for {dataset}')
        for stage, command in [('preparation', prepare), ('pipeline', pipeline)]:
            print(json.dumps({'dataset': dataset, 'stage': stage, 'status': 'started',
                              'physical_gpu': args.gpu}), flush=True)
            started = time.perf_counter()
            with (destination / f'{dataset}_{stage}.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               cwd=HERE.parent, check=True)
            print(json.dumps({'dataset': dataset, 'stage': stage, 'status': 'complete',
                              'wall_seconds': time.perf_counter() - started}), flush=True)
        result_path = destination / dataset / 'timing.json'
        result = json.loads(result_path.read_text())
        result['interpretation'] = f'Timing-only proxy using a {args.steps}-update unconverged backbone; not trained-model quality or speed'
        result['preparation_excluded_from_pipeline'] = True
        result['preparation_command'] = prepare
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
