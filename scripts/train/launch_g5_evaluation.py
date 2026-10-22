"""Evaluate H1 and all G5 arms on identical Development caches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
G5 = ROOT / 'results/stage3/g5_object_dynamics'
G3_CACHE = ROOT / 'results/stage3/g3_history/shared_eval_cache'
OBJECT_CACHE = ROOT / 'results/stage3/g4_diagnostics/object_readout_cache_postnorm'
H1 = ROOT / 'results/stage3/g3_history/H1_uniform_mse/checkpoint_step2000.pt'
ARMS = ('A_uniform_control', 'B_motion_balanced', 'C_action_sensitive', 'D_combined')


def run(command, log):
    began = time.perf_counter()
    with log.open('xb') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return time.perf_counter() - began


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--g5-root', type=Path, default=G5)
    parser.add_argument('--g3-cache', type=Path, default=G3_CACHE)
    parser.add_argument('--object-cache', type=Path, default=OBJECT_CACHE)
    parser.add_argument('--h1-checkpoint', type=Path, default=H1)
    args = parser.parse_args(argv)
    g5 = args.g5_root.resolve()
    g3_cache = args.g3_cache.resolve()
    object_cache = args.object_cache.resolve()
    h1 = args.h1_checkpoint.resolve()
    if (g5 / 'STATUS').read_text().strip() != 'TRAINING_DONE_EVALUATION_PENDING':
        raise RuntimeError('G5 training is not ready for evaluation')
    output = g5 / 'evaluation'
    output.mkdir(exist_ok=False)
    checkpoints = {'H1_source': h1}
    checkpoints.update({arm: g5 / arm / 'checkpoint_step2500.pt' for arm in ARMS})
    report = {'cohort': 'G3 Development; final holdout not used', 'runs': []}
    (g5 / 'STATUS').write_text('EVALUATION_RUNNING\n')
    for name, checkpoint in checkpoints.items():
        common = output / f'{name}_common.json'
        obj = output / f'{name}_object_postnorm.json'
        commands = [
            [
                '/venv/stable-wm/bin/python', str(ROOT / 'scripts/train/g3_evaluate.py'),
                '--checkpoint', str(checkpoint), '--cache-dir', str(g3_cache),
                '--output', str(common), '--history', '1',
            ],
            [
                '/venv/stable-wm/bin/python',
                str(ROOT / 'scripts/train/g4_object_readout_evaluate.py'),
                '--checkpoint', str(checkpoint), '--g3-cache', str(g3_cache),
                '--object-cache', str(object_cache), '--output', str(obj),
                '--history', '1',
            ],
        ]
        for kind, command in zip(('common', 'object_postnorm'), commands):
            seconds = run(command, output / f'{name}_{kind}.log')
            report['runs'].append(
                {'model': name, 'kind': kind, 'seconds': seconds, 'command': command}
            )
            (output / 'launcher.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'STATUS').write_text('READY\n')
    (g5 / 'STATUS').write_text('EVALUATION_DONE_ANALYSIS_PENDING\n')


if __name__ == '__main__':
    main()
