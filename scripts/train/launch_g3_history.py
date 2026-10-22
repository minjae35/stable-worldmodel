"""Prepare common anchors and sequentially launch G3 H1/H3 training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/stable-worldmodel')
SOURCE_MANIFEST = ROOT / 'results/stage3/g2_c1_followup/manifest.json'
NORMALIZERS = ROOT / 'results/stage3/g2_fresh_ab/action_normalizers.json'

sys.path.insert(0, str(ROOT))
from scripts.train.g2_spatial import sha256_file  # noqa: E402
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES  # noqa: E402
from stable_worldmodel.wm.spatial.g3_history import (  # noqa: E402
    common_history_items,
    items_sha256,
)


def build_common_manifest() -> dict:
    source = json.loads(SOURCE_MANIFEST.read_text())
    frameskip = int(source['frameskip'])
    environments = {}
    for env in ENV_NAMES:
        environments[env] = {}
        for split in ('train', 'val', 'test'):
            selected = common_history_items(
                source[env]['s0'][split], frameskip, history=3
            )
            if set(item['episode'] for item in selected) != set(source[env]['split'][split]):
                raise ValueError(f'{env}/{split} lost an episode after history filtering')
            environments[env][split] = selected
            environments[env][f'{split}_sha256'] = items_sha256(selected)
    return {
        'schema': 1,
        'purpose': 'G3 H1/H3 common history-valid target anchors',
        'source_manifest': str(SOURCE_MANIFEST),
        'source_manifest_sha256': sha256_file(SOURCE_MANIFEST),
        'sampling': 'source s0 uniform anchors filtered only by start >= 10',
        'frameskip': frameskip,
        'max_history': 3,
        'alignment': {
            'H1': {'observations': ['t'], 'actions': ['[t,t+5)'], 'target': 't+5'},
            'H3': {
                'observations': ['t-10', 't-5', 't'],
                'actions': ['[t-10,t-5)', '[t-5,t)', '[t,t+5)'],
                'target': 't+5',
            },
            'target_anchor_shared': True,
            'future_information_used': False,
        },
        'training_columns': ['episode_idx', 'pixels', 'action'],
        'forbidden_training_inputs': source['forbidden_in_sampling_and_loss'],
        'environments': environments,
    }


def build_commands(python: str, output: Path, steps: int, checkpoint_every: int, max_items: int):
    common = output / 'common_windows.json'
    commands = []
    for arm, name in (('H1', 'H1_uniform_mse'), ('H3', 'H3_uniform_mse')):
        command = [
            python, str(ROOT / 'scripts/train/g3_history_spatial.py'),
            '--arm', arm,
            '--output-dir', str(output / name),
            '--common-manifest', str(common),
            '--normalizer-file', str(NORMALIZERS),
            '--steps', str(steps),
            '--checkpoint-every', str(checkpoint_every),
        ]
        if max_items > 0:
            command.extend(['--max-train-items', str(max_items)])
        commands.append(command)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--python', default='/venv/stable-wm/bin/python')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--max-train-items', type=int, default=0)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reuse-prepared-root', action='store_true')
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not SOURCE_MANIFEST.is_file() or not NORMALIZERS.is_file():
        raise FileNotFoundError('source manifest or action normalizers are missing')
    common = build_common_manifest()
    if output.exists():
        if not args.reuse_prepared_root:
            raise FileExistsError(f'refusing to overwrite {output}')
        if (output / 'STATUS').read_text().strip() != 'PREPARED':
            raise RuntimeError('existing output root is not in PREPARED state')
        if json.loads((output / 'common_windows.json').read_text()) != common:
            raise ValueError('prepared common manifest no longer matches source')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'common_windows.json').write_text(json.dumps(common, indent=2) + '\n')
        (output / 'STATUS').write_text('PREPARED\n')
    if args.prepare_only:
        return
    commands = build_commands(
        args.python, output, args.steps, args.checkpoint_every, args.max_train_items
    )
    report = {
        'sequential': True,
        'stop_on_first_failure': True,
        'commands': commands,
        'runs': [],
    }
    config_path = output / 'launcher_config.json'
    config_path.write_text(json.dumps(report, indent=2) + '\n')
    (output / 'STATUS').write_text('RUNNING\n')
    for command in commands:
        arm = command[command.index('--arm') + 1]
        log_path = output / f'{arm}_training.log'
        began = time.perf_counter()
        with log_path.open('xb') as handle:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert process.stdout is not None
            for chunk in iter(process.stdout.readline, b''):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                handle.write(chunk)
                handle.flush()
            returncode = process.wait()
        report['runs'].append({
            'arm': arm,
            'returncode': returncode,
            'seconds': time.perf_counter() - began,
            'log': str(log_path),
        })
        config_path.write_text(json.dumps(report, indent=2) + '\n')
        if returncode:
            (output / 'STATUS').write_text(f'FAILED arm={arm} returncode={returncode}\n')
            raise SystemExit(returncode)
    (output / 'STATUS').write_text('TRAINING_DONE_EVALUATION_PENDING\n')


if __name__ == '__main__':
    main()
