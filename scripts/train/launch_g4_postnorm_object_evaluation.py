"""Evaluate H1/H3/G4 with the validated post-norm object-local probe.

This launcher intentionally writes to a new output tree.  It never reuses the
superseded pre-norm object-readout results.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
G3 = ROOT / 'results/stage3/g3_history'
G4 = ROOT / 'results/stage3/g4_diagnostics'
CACHE = G3 / 'shared_eval_cache'
OBJECT = G4 / 'object_readout_cache_postnorm'
OUTPUT = G4 / 'object_readout_evaluations_postnorm'
STEPS = (500, 1000, 1500, 2000)


def main() -> None:
    if (OBJECT / 'STATUS').read_text().strip() != 'READY':
        raise RuntimeError(f'post-norm object cache is not READY: {OBJECT}')
    if OUTPUT.exists():
        raise FileExistsError(f'refusing to overwrite {OUTPUT}')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    launcher_path = OUTPUT / 'launcher.json'
    report = {'status': 'RUNNING', 'jobs': []}
    launcher_path.write_text(json.dumps(report, indent=2) + '\n')
    python = '/venv/stable-wm/bin/python'
    for arm, directory, history, temporal in (
        ('H1', G3 / 'H1_uniform_mse', '1', 'none'),
        ('H3', G3 / 'H3_uniform_mse', '3', 'none'),
        ('G4', G4 / 'H3_sinusoidal', '3', 'sinusoidal'),
    ):
        arm_dir = OUTPUT / arm
        arm_dir.mkdir()
        for step in STEPS:
            output = arm_dir / f'step{step}.json'
            command = [
                python, str(ROOT / 'scripts/train/g4_object_readout_evaluate.py'),
                '--checkpoint', str(directory / f'checkpoint_step{step}.pt'),
                '--g3-cache', str(CACHE), '--object-cache', str(OBJECT),
                '--output', str(output), '--history', history,
                '--temporal-encoding', temporal,
            ]
            began = time.perf_counter()
            result = subprocess.run(command)
            job = {
                'arm': arm, 'step': step, 'output': str(output),
                'returncode': result.returncode,
                'seconds': time.perf_counter() - began,
                'command': command,
            }
            report['jobs'].append(job)
            launcher_path.write_text(json.dumps(report, indent=2) + '\n')
            if result.returncode:
                report['status'] = 'FAILED'
                launcher_path.write_text(json.dumps(report, indent=2) + '\n')
                raise SystemExit(result.returncode)
    report['status'] = 'DONE'
    launcher_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
