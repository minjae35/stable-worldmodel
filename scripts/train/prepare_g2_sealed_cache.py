"""Encode the pre-frozen G2 Fresh sealed cohort without fitting new probes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import lance
import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_evaluate_shared import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    canonical_hash,
    dataset_identity,
)
from scripts.train.g2_spatial import (  # noqa: E402
    DATASETS,
    DINO_WEIGHTS,
    encode_images,
    normalize_actions,
    sha256_file,
)
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES, load_label_free_clips  # noqa: E402
from stable_worldmodel.wm.spatial.g2_eval import load_state_labels, pool4  # noqa: E402
from stable_worldmodel.wm.spatial.g2_fresh import torch_normalizers  # noqa: E402
from stable_worldmodel.wm.spatial.model import FrozenDINOv2  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-manifest', type=Path, required=True)
    parser.add_argument('--source-probe-cache', type=Path, required=True)
    parser.add_argument('--action-normalizers', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--encode-batch', type=int, default=32)
    args = parser.parse_args()

    cohort_path = args.cohort_manifest.resolve()
    probe_cache = args.source_probe_cache.resolve()
    normalizer_path = args.action_normalizers.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    cohort = json.loads(cohort_path.read_text())
    if not cohort.get('frozen_before_training'):
        raise ValueError('sealed cohort was not marked frozen before training')
    normalizer_artifact = json.loads(normalizer_path.read_text())
    normalizers = torch_normalizers(normalizer_artifact)
    source_probe_report = json.loads((probe_cache / 'probes.json').read_text())

    identity = {
        'schema': 1,
        'purpose': cohort['purpose'],
        'cohort_manifest_sha256': sha256_file(cohort_path),
        'dino_weights_sha256': sha256_file(DINO_WEIGHTS / 'model.safetensors'),
        'action_normalizer_key': normalizer_artifact['key'],
        'source_probe_key': source_probe_report['probe_key'],
        'probe_training_data': 'development train/validation only; sealed test unused',
        'limits': {'test': 0},
        'preprocessing': {
            'image_size': [224, 224],
            'rgb': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'token_shape': [256, 384],
            'pooling': 'adaptive_avg_pool2d_4x4',
        },
        'datasets': {env: dataset_identity(DATASETS[env]) for env in ENV_NAMES},
    }
    metadata = {'cache_key': canonical_hash(identity), 'identity': identity, 'timing': {}}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'STATUS').write_text('BUILDING\n')
    device = torch.device('cuda:0')
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device)
    frameskip = int(cohort['frameskip'])
    for env in ENV_NAMES:
        dataset = lance.dataset(str(DATASETS[env]))
        clips = list(cohort['environments'][env]['clips'])
        began = time.perf_counter()
        raw = load_label_free_clips(dataset, clips, env, frameskip)
        labels = load_state_labels(dataset, clips, env, frameskip)
        metadata['timing'][f'{env}/lance_decode_s'] = time.perf_counter() - began
        began = time.perf_counter()
        context = encode_images(encoder, raw['images_context'], device, args.encode_batch)
        target = encode_images(encoder, raw['images_target'], device, args.encode_batch)
        metadata['timing'][f'{env}/dino_encode_s'] = time.perf_counter() - began
        shard = {
            'context': context,
            'target': target,
            'context_pool4': torch.from_numpy(pool4(context)),
            'target_pool4': torch.from_numpy(pool4(target)),
            'actions': normalize_actions(raw['actions'], normalizers[env]),
            'episode': torch.as_tensor(raw['episode'], dtype=torch.long),
            'start': torch.as_tensor(raw['start'], dtype=torch.long),
            'labels': {
                name: {
                    'current': torch.from_numpy(current),
                    'future': torch.from_numpy(future),
                }
                for name, (current, future) in labels.items()
            },
        }
        atomic_torch_save(shard, output / f'{env}_test.pt')
    atomic_json(metadata, output / 'metadata.json')
    shutil.copy2(probe_cache / 'probes.pt', output / 'probes.pt')
    shutil.copy2(probe_cache / 'probes.json', output / 'probes.json')
    (output / 'STATUS').write_text('READY\n')
    print(json.dumps({
        'output': str(output),
        'cache_key': metadata['cache_key'],
        'source_probe_key': source_probe_report['probe_key'],
        'timing': metadata['timing'],
    }, indent=2))


if __name__ == '__main__':
    main()
