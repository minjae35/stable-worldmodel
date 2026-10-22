"""Inventory production G0-G5 binary artifacts without modifying them."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path('/workspace/stable-worldmodel')
RESULTS = ROOT / 'results/stage3'
OUTPUT = ROOT / 'reproducibility'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stage(path: Path) -> str:
    first = path.relative_to(RESULTS).parts[0]
    return first.split('_', 1)[0].upper()


def experiment(path: Path) -> str:
    relative = path.relative_to(RESULTS)
    return '/'.join(relative.parts[:-1])


def inspect_checkpoint(path: Path) -> dict:
    relative = path.relative_to(ROOT).as_posix()
    row = {
        'stage': stage(path),
        'experiment': experiment(path),
        'path': relative,
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
        'step': '',
        'model_state': False,
        'optimizer_state': False,
        'scheduler_state': False,
        'rng_state': False,
        'config_hash': False,
        'resume_class': '',
    }
    if path.suffix == '.safetensors':
        row['resume_class'] = 'frozen_pretrained_backbone'
        return row
    payload = torch.load(path, map_location='cpu', weights_only=True)
    row['step'] = int(payload.get('step', -1))
    row['model_state'] = 'model' in payload
    row['optimizer_state'] = 'optimizer' in payload
    row['scheduler_state'] = any(
        key in payload for key in ('scheduler', 'lr_scheduler', 'scheduler_state_dict')
    )
    extra = payload.get('extra', {})
    row['rng_state'] = 'rng_state' in payload or (
        isinstance(extra, dict) and 'rng_state' in extra
    )
    row['config_hash'] = 'config_sha256' in payload
    if row['model_state'] and row['optimizer_state'] and row['rng_state']:
        row['resume_class'] = 'exact_model_optimizer_rng'
    elif row['model_state'] and row['optimizer_state']:
        row['resume_class'] = 'model_optimizer_no_rng'
    elif row['model_state']:
        row['resume_class'] = 'model_weights_only'
    else:
        row['resume_class'] = 'unknown_payload'
    del payload
    return row


def candidates() -> list[Path]:
    paths = [RESULTS / 'g0_dinov2/weights/model.safetensors']
    for path in RESULTS.rglob('checkpoint_step*.pt'):
        relative = path.relative_to(RESULTS).as_posix().lower()
        if 'smoke' in relative or '/cache/' in relative:
            continue
        paths.append(path)
    return sorted(set(paths), key=lambda value: value.relative_to(ROOT).as_posix())


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    rows = [inspect_checkpoint(path) for path in candidates()]
    row_by_path = {row['path']: row for row in rows}
    all_checkpoint_paths = sorted(
        RESULTS.rglob('checkpoint_step*.pt'),
        key=lambda value: value.relative_to(ROOT).as_posix(),
    )
    all_checkpoint_rows = []
    for path in all_checkpoint_paths:
        relative = path.relative_to(ROOT).as_posix()
        row = dict(row_by_path.get(relative) or inspect_checkpoint(path))
        lowered = relative.lower()
        row['backup_class'] = (
            'regenerable_smoke' if 'smoke' in lowered else 'required_production'
        )
        all_checkpoint_rows.append(row)
    all_fields = list(all_checkpoint_rows[0])
    with (OUTPUT / 'STAGE3_ALL_CHECKPOINTS.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, all_fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(all_checkpoint_rows)
    manifest = OUTPUT / 'STAGE3_CHECKPOINT_MANIFEST.tsv'
    fields = list(rows[0])
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    required = OUTPUT / 'STAGE3_CHECKPOINTS_REQUIRED.txt'
    required.write_text(''.join(f'{row["path"]}\n' for row in rows))
    classes: dict[str, int] = {}
    for row in rows:
        classes[row['resume_class']] = classes.get(row['resume_class'], 0) + 1
    summary = {
        'scope': 'G0 frozen encoder plus production G1-G5 checkpoints',
        'excludes': ['smoke checkpoints', 'feature/probe caches', 'logs', 'datasets'],
        'file_count': len(rows),
        'total_bytes': sum(int(row['bytes']) for row in rows),
        'total_gib': sum(int(row['bytes']) for row in rows) / 1024**3,
        'resume_class_counts': classes,
        'all_checkpoint_files': len(all_checkpoint_rows),
        'all_checkpoint_bytes': sum(int(row['bytes']) for row in all_checkpoint_rows),
        'regenerable_smoke_checkpoint_files': sum(
            row['backup_class'] == 'regenerable_smoke' for row in all_checkpoint_rows
        ),
        'regenerable_smoke_checkpoint_bytes': sum(
            int(row['bytes']) for row in all_checkpoint_rows
            if row['backup_class'] == 'regenerable_smoke'
        ),
        'manifest': manifest.relative_to(ROOT).as_posix(),
        'rsync_files_from': required.relative_to(ROOT).as_posix(),
    }
    (OUTPUT / 'STAGE3_CHECKPOINT_SUMMARY.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )
    evaluation_paths = [
        RESULTS / 'g2_2x2/shared_eval_cache/probes.pt',
        RESULTS / 'g3_history/shared_eval_cache/pusht_test.pt',
        RESULTS / 'g3_history/shared_eval_cache/tworoom_test.pt',
        RESULTS / 'g3_history/shared_eval_cache/cube_test.pt',
        RESULTS / 'g4_diagnostics/representation_postnorm/selected_probes.pt',
        RESULTS / 'g4_diagnostics/object_readout_cache_postnorm/pusht_object_readout.pt',
    ]
    evaluation_rows = [
        {
            'path': path.relative_to(ROOT).as_posix(),
            'bytes': path.stat().st_size,
            'sha256': sha256_file(path),
            'role': 'exact_evaluation_cache_or_probe',
        }
        for path in evaluation_paths
    ]
    with (OUTPUT / 'STAGE3_EVALUATION_ARTIFACTS.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            evaluation_rows[0],
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(evaluation_rows)
    (OUTPUT / 'STAGE3_EVALUATION_ARTIFACTS_RECOMMENDED.txt').write_text(
        ''.join(f'{row["path"]}\n' for row in evaluation_rows)
    )
    summary['recommended_evaluation_artifacts'] = {
        'file_count': len(evaluation_rows),
        'total_bytes': sum(row['bytes'] for row in evaluation_rows),
        'total_gib': sum(row['bytes'] for row in evaluation_rows) / 1024**3,
        'note': 'Regenerable, but required to reproduce evaluations byte-for-byte.',
    }
    (OUTPUT / 'STAGE3_CHECKPOINT_SUMMARY.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
