"""Two-phase sequential launcher: validate all smoke prefixes, then finish G5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from stable_worldmodel.wm.spatial.g5 import SOURCE_STEP  # noqa: E402
from stable_worldmodel.wm.spatial.g5_objective import G5_ARMS  # noqa: E402


def command_for(python: str, root: Path, arm: str, resume: bool) -> list[str]:
    command = [
        python, str(ROOT / 'scripts/train/g5_object_dynamics.py'),
        '--arm', arm, '--output-dir', str(root / arm),
        '--plan', str(root / 'experiment_plan.json'),
    ]
    if resume:
        command.extend(['--resume', str(root / arm / f'checkpoint_step{SOURCE_STEP + 10}.pt')])
    else:
        command.extend(['--stop-after-update', '10'])
    return command


def validate_smoke(root: Path, arm: str) -> dict:
    run = root / arm
    if (run / 'STATUS').read_text().strip() != 'PAUSED_AFTER_VALIDATED_SMOKE':
        raise RuntimeError(f'{arm} did not pause cleanly after smoke')
    rows = list(csv.DictReader((run / 'metrics.csv').open()))
    if len(rows) != 10 or int(rows[-1]['new_update']) != 10:
        raise RuntimeError(f'{arm} smoke metrics are not exactly 10 updates')
    for row in rows:
        for key, value in row.items():
            if key not in ('new_update', 'cumulative_step') and not math.isfinite(float(value)):
                raise RuntimeError(f'{arm} has non-finite {key} in smoke')
    checkpoint = run / f'checkpoint_step{SOURCE_STEP + 10}.pt'
    if not checkpoint.is_file():
        raise RuntimeError(f'{arm} smoke checkpoint is missing')
    return {
        'updates': 10, 'checkpoint': str(checkpoint),
        'last_objective': float(rows[-1]['objective']),
        'peak_vram_gb': max(float(row['peak_vram_gb']) for row in rows),
    }


def run_logged(command, log_path):
    began = time.perf_counter()
    with log_path.open('xb') as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for chunk in iter(process.stdout.readline, b''):
            sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            handle.write(chunk); handle.flush()
        returncode = process.wait()
    return returncode, time.perf_counter() - began


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--python', default='/venv/stable-wm/bin/python')
    args = parser.parse_args()
    root = args.output_root.resolve()
    if (root / 'STATUS').read_text().strip() != 'PREPARED':
        raise RuntimeError('G5 root is not in PREPARED state')
    plan = json.loads((root / 'experiment_plan.json').read_text())
    if plan['arms'] != list(G5_ARMS) or plan['total_optimizer_updates'] != 2000:
        raise ValueError('G5 plan does not encode the exact four-arm 2000-update budget')
    report = {'sequential': True, 'stop_on_first_failure': True, 'phases': []}
    launcher = root / 'launcher.json'
    launcher.write_text(json.dumps(report, indent=2) + '\n')
    (root / 'STATUS').write_text('SMOKE_RUNNING\n')
    smoke_phase = {'name': 'smoke_prefix_in_budget', 'runs': []}
    for arm in G5_ARMS:
        command = command_for(args.python, root, arm, False)
        code, seconds = run_logged(command, root / f'{arm}_smoke.log')
        smoke_phase['runs'].append(
            {'arm': arm, 'command': command, 'returncode': code, 'seconds': seconds}
        )
        report['phases'] = [smoke_phase]
        launcher.write_text(json.dumps(report, indent=2) + '\n')
        if code:
            (root / 'STATUS').write_text(f'FAILED_SMOKE arm={arm} returncode={code}\n')
            raise SystemExit(code)
    smoke_phase['validation'] = {arm: validate_smoke(root, arm) for arm in G5_ARMS}
    launcher.write_text(json.dumps(report, indent=2) + '\n')
    (root / 'STATUS').write_text('SMOKE_VALIDATED\n')

    train_phase = {'name': 'resume_to_500', 'runs': []}
    report['phases'].append(train_phase)
    (root / 'STATUS').write_text('TRAINING_RUNNING\n')
    for arm in G5_ARMS:
        command = command_for(args.python, root, arm, True)
        code, seconds = run_logged(command, root / f'{arm}_training.log')
        train_phase['runs'].append(
            {'arm': arm, 'command': command, 'returncode': code, 'seconds': seconds}
        )
        launcher.write_text(json.dumps(report, indent=2) + '\n')
        if code:
            (root / 'STATUS').write_text(f'FAILED_TRAINING arm={arm} returncode={code}\n')
            raise SystemExit(code)
    (root / 'STATUS').write_text('TRAINING_DONE_EVALUATION_PENDING\n')


if __name__ == '__main__':
    main()
