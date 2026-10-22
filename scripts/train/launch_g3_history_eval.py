"""Sequential common evaluation and bootstrap launcher for G3 H1/H3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--python', default='/venv/stable-wm/bin/python')
    parser.add_argument('--steps', type=int, nargs='+', default=(500, 1000, 1500, 2000))
    args = parser.parse_args()
    root = args.output_root.resolve()
    cache = root / 'shared_eval_cache'
    if (cache / 'STATUS').read_text().strip() != 'READY':
        raise RuntimeError('G3 evaluation cache is not READY')
    status = root / 'EVALUATION_STATUS'
    if status.exists():
        raise FileExistsError('evaluation status already exists; refusing overwrite')
    jobs = []
    for step in args.steps:
        for arm, history, directory in (
            ('H1', 1, 'H1_uniform_mse'),
            ('H3', 3, 'H3_uniform_mse'),
        ):
            checkpoint = root / directory / f'checkpoint_step{step}.pt'
            output = root / directory / f'evaluation_development_step{step}.json'
            jobs.append({
                'kind': 'evaluation', 'arm': arm, 'step': step, 'output': str(output),
                'command': [
                    args.python, str(ROOT / 'scripts/train/g3_evaluate.py'),
                    '--checkpoint', str(checkpoint), '--cache-dir', str(cache),
                    '--output', str(output), '--history', str(history),
                ],
            })
        jobs.append({
            'kind': 'paired_bootstrap', 'step': step,
            'output': str(root / f'development_paired_bootstrap_step{step}.json'),
            'command': [
                args.python, str(ROOT / 'scripts/train/compare_g3_history.py'),
                '--checkpoint-h1', str(root / 'H1_uniform_mse' / f'checkpoint_step{step}.pt'),
                '--checkpoint-h3', str(root / 'H3_uniform_mse' / f'checkpoint_step{step}.pt'),
                '--cache-dir', str(cache),
                '--output', str(root / f'development_paired_bootstrap_step{step}.json'),
            ],
        })
    missing = []
    for job in jobs:
        command = job['command']
        for flag in ('--checkpoint', '--checkpoint-h1', '--checkpoint-h3'):
            if flag in command:
                value = Path(command[command.index(flag) + 1])
                if not value.is_file():
                    missing.append(str(value))
    if missing:
        raise FileNotFoundError(f'missing checkpoints: {missing}')
    report = {'jobs': jobs, 'completed': []}
    config = root / 'evaluation_launcher_config.json'
    config.write_text(json.dumps(report, indent=2) + '\n')
    status.write_text('RUNNING\n')
    try:
        for job in jobs:
            destination = Path(job['output'])
            if destination.exists():
                raise FileExistsError(f'refusing to overwrite {destination}')
            began = time.perf_counter()
            result = subprocess.run(job['command'], check=False)
            record = {
                'kind': job['kind'], 'step': job['step'],
                'arm': job.get('arm'), 'returncode': result.returncode,
                'seconds': time.perf_counter() - began,
                'output': job['output'],
            }
            report['completed'].append(record)
            config.write_text(json.dumps(report, indent=2) + '\n')
            if result.returncode:
                raise RuntimeError(f'evaluation job failed: {record}')
        status.write_text('DONE\n')
    except BaseException as error:
        status.write_text(f'FAILED {type(error).__name__}: {error}\n')
        raise


if __name__ == '__main__':
    main()
