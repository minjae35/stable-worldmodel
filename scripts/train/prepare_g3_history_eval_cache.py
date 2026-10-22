"""Prepare the shared G3 H1/H3 development cache on common target anchors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import lance
import torch

ROOT = Path('/workspace/stable-worldmodel')
sys.path.insert(0, str(ROOT))

from scripts.train.g2_evaluate_shared import canonical_hash  # noqa: E402
from scripts.train.g2_spatial import (  # noqa: E402
    DATASETS,
    DINO_WEIGHTS,
    MANIFEST,
    encode_images,
    sha256_file,
)
from stable_worldmodel.wm.spatial.g2 import ENV_NAMES  # noqa: E402
from stable_worldmodel.wm.spatial.g2_fresh import torch_normalizers  # noqa: E402
from stable_worldmodel.wm.spatial.g3_history import (  # noqa: E402
    load_label_free_history_clips,
    normalize_action_history,
)
from stable_worldmodel.wm.spatial.model import FrozenDINOv2  # noqa: E402

BASE_CACHE = ROOT / 'results/stage3/g2_2x2/shared_eval_cache'
NORMALIZERS = ROOT / 'results/stage3/g2_fresh_ab/action_normalizers.json'


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--encode-batch', type=int, default=32)
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    if cache.exists():
        raise FileExistsError(f'refusing to overwrite {cache}')
    if (BASE_CACHE / 'STATUS').read_text().strip() != 'READY':
        raise RuntimeError('base G2 evaluation cache is not READY')
    common_path = args.common_manifest.resolve()
    common = json.loads(common_path.read_text())
    source = json.loads(MANIFEST.read_text())
    base_metadata = json.loads((BASE_CACHE / 'metadata.json').read_text())
    probe_metadata = json.loads((BASE_CACHE / 'probes.json').read_text())
    normalizer_artifact = json.loads(NORMALIZERS.read_text())
    normalizers = torch_normalizers(normalizer_artifact)
    identity = {
        'schema': 1,
        'purpose': 'G3 shared H1/H3 development evaluation cache',
        'common_manifest_sha256': sha256_file(common_path),
        'source_manifest_sha256': sha256_file(MANIFEST),
        'base_feature_cache_key': base_metadata['cache_key'],
        'reused_probe_key': probe_metadata['probe_key'],
        'dino_weights_sha256': sha256_file(DINO_WEIGHTS / 'model.safetensors'),
        'normalizer_sha256': sha256_file(NORMALIZERS),
        'history': 3,
        'frameskip': 5,
        'preprocessing': base_metadata['identity']['preprocessing'],
        'cohort': {
            env: common['environments'][env]['test_sha256'] for env in ENV_NAMES
        },
    }
    metadata = {'cache_key': canonical_hash(identity), 'identity': identity, 'counts': {}}
    cache.mkdir(parents=True, exist_ok=False)
    (cache / 'STATUS').write_text('BUILDING\n')
    device = torch.device('cuda:0')
    encoder = FrozenDINOv2.from_pretrained(str(DINO_WEIGHTS)).to(device)

    for env in ENV_NAMES:
        source_items = source[env]['s0']['test']
        source_index = {
            (int(item['episode']), int(item['start'])): index
            for index, item in enumerate(source_items)
        }
        items = common['environments'][env]['test']
        indices = torch.as_tensor([
            source_index[(int(item['episode']), int(item['start']))]
            for item in items
        ], dtype=torch.long)
        base = torch.load(
            BASE_CACHE / f'{env}_test.pt', map_location='cpu', weights_only=True
        )
        raw = load_label_free_history_clips(
            lance.dataset(str(DATASETS[env])), items, env, 5, 3
        )
        if not torch.equal(torch.as_tensor(raw['episode']), base['episode'][indices]):
            raise RuntimeError(f'{env} episode alignment mismatch')
        if not torch.equal(torch.as_tensor(raw['start']), base['start'][indices]):
            raise RuntimeError(f'{env} start alignment mismatch')
        older = raw['images_context'][:, :2]
        encoded_older = encode_images(
            encoder,
            older.reshape(-1, *older.shape[2:]),
            device,
            args.encode_batch,
        ).reshape(len(items), 2, 256, 384)
        current = base['context'].index_select(0, indices)[:, None]
        labels = {
            name: {
                'current': values['current'].index_select(0, indices),
                'future': values['future'].index_select(0, indices),
            }
            for name, values in base['labels'].items()
        }
        shard = {
            'context': torch.cat([encoded_older, current], dim=1),
            'target': base['target'].index_select(0, indices),
            'context_pool4': base['context_pool4'].index_select(0, indices),
            'target_pool4': base['target_pool4'].index_select(0, indices),
            'actions': normalize_action_history(raw['actions'], normalizers[env]),
            'episode': base['episode'].index_select(0, indices),
            'start': base['start'].index_select(0, indices),
            'labels': labels,
            'source_test_indices': indices,
        }
        atomic_torch_save(shard, cache / f'{env}_test.pt')
        metadata['counts'][env] = len(items)
    del encoder
    torch.cuda.empty_cache()
    (cache / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (cache / 'STATUS').write_text('READY\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
