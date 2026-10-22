"""Run common G3/G4 evaluation, object-local readout, and paired bootstraps."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
G3 = ROOT / 'results/stage3/g3_history'
G4 = ROOT / 'results/stage3/g4_diagnostics'
CACHE = G3 / 'shared_eval_cache'
OBJECT = G4 / 'object_readout_cache'
STEPS = (500, 1000, 1500, 2000)


def run_job(job: dict) -> dict:
    if Path(job['output']).exists():
        return {**job, 'status': 'REUSED_EXISTING', 'seconds': 0.0}
    began = time.perf_counter()
    result = subprocess.run(job['command'])
    return {
        **job,
        'status': 'DONE' if result.returncode == 0 else 'FAILED',
        'returncode': result.returncode,
        'seconds': time.perf_counter() - began,
    }


def main() -> None:
    jobs = []
    python = '/venv/stable-wm/bin/python'
    for step in STEPS:
        output = G4 / 'H3_sinusoidal' / f'evaluation_development_step{step}.json'
        jobs.append({
            'kind': 'standard_evaluation', 'arm': 'G4', 'step': step, 'output': str(output),
            'command': [
                python, str(ROOT / 'scripts/train/g3_evaluate.py'),
                '--checkpoint', str(G4 / f'H3_sinusoidal/checkpoint_step{step}.pt'),
                '--cache-dir', str(CACHE), '--output', str(output), '--history', '3',
                '--temporal-encoding', 'sinusoidal', '--temporal-encoding-scale', '1.0',
            ],
        })
    for arm, directory, temporal in (
        ('H1', G3 / 'H1_uniform_mse', 'none'),
        ('H3', G3 / 'H3_uniform_mse', 'none'),
        ('G4', G4 / 'H3_sinusoidal', 'sinusoidal'),
    ):
        history = '1' if arm == 'H1' else '3'
        for step in STEPS:
            output = G4 / 'object_readout_evaluations' / arm / f'step{step}.json'
            if arm == 'G4' and step == 500:
                # Preserve the validated first evaluation and copy neither data nor results.
                output = G4 / 'H3_sinusoidal/object_readout_development_step500.json'
            jobs.append({
                'kind': 'object_readout', 'arm': arm, 'step': step, 'output': str(output),
                'command': [
                    python, str(ROOT / 'scripts/train/g4_object_readout_evaluate.py'),
                    '--checkpoint', str(directory / f'checkpoint_step{step}.pt'),
                    '--g3-cache', str(CACHE), '--object-cache', str(OBJECT),
                    '--output', str(output), '--history', history,
                    '--temporal-encoding', temporal,
                ],
            })
    for step in STEPS:
        output = G4 / f'paired_bootstrap_G4_minus_H3_step{step}.json'
        jobs.append({
            'kind': 'paired_bootstrap', 'arm': 'G4-H3', 'step': step, 'output': str(output),
            'command': [
                python, str(ROOT / 'scripts/train/compare_g4_temporal.py'),
                '--baseline', str(G3 / f'H3_uniform_mse/checkpoint_step{step}.pt'),
                '--experimental', str(G4 / f'H3_sinusoidal/checkpoint_step{step}.pt'),
                '--cache-dir', str(CACHE), '--output', str(output),
            ],
        })
    report = {'jobs': jobs, 'completed': [], 'status': 'RUNNING'}
    path = G4 / 'evaluation_launcher.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    for job in jobs:
        completed = run_job(job)
        report['completed'].append(completed)
        path.write_text(json.dumps(report, indent=2) + '\n')
        if completed.get('returncode', 0):
            report['status'] = 'FAILED'
            path.write_text(json.dumps(report, indent=2) + '\n')
            raise SystemExit(completed['returncode'])
    report['status'] = 'DONE'
    path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
