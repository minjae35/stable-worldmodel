"""Freeze an episode-disjoint sealed cohort before G2 Fresh A/B training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lance
import numpy as np

ROOT = Path('/workspace/stable-worldmodel')
MANIFEST = ROOT / 'results/stage3/g2_c1_followup/manifest.json'
DATASETS = {
    'pusht': Path('/workspace/datasets/lewm-pusht/train.lance'),
    'tworoom': Path('/workspace/datasets/lewm-tworooms/train.lance'),
    'cube': Path('/workspace/datasets/lewm-cube/train.lance'),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def episode_rows(dataset) -> dict[int, tuple[int, int]]:
    values = np.asarray(dataset.to_table(columns=['episode_idx'])['episode_idx'].to_numpy())
    boundaries = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
    result = {}
    for first, stop in zip(boundaries[:-1], boundaries[1:]):
        episode = int(values[first])
        if episode in result:
            raise ValueError(f'episode {episode} is not contiguous')
        result[episode] = (int(first), int(stop - first))
    return result


def build_sealed(source: dict, n_episodes: int, seed: int) -> dict:
    document = {
        'schema': 1,
        'purpose': 'sealed_test_not_for_pilot_hyperparameter_selection',
        'frozen_before_training': True,
        'source_manifest': str(MANIFEST),
        'source_manifest_sha256': sha256_file(MANIFEST),
        'selection_seed': seed,
        'frameskip': int(source['frameskip']),
        'span': int(source['span']),
        'clips_per_episode': 4,
        'environments': {},
    }
    span = int(source['span'])
    for env_index, (env, path) in enumerate(DATASETS.items()):
        dataset = lance.dataset(str(path))
        rows = episode_rows(dataset)
        used = set().union(*(
            {int(value) for value in source[env]['split'][name]}
            for name in ('train', 'val', 'test')
        ))
        eligible = sorted(
            episode for episode, (_, length) in rows.items()
            if episode not in used and length >= span + 1
        )
        if len(eligible) < n_episodes:
            raise ValueError(f'{env} has only {len(eligible)} unused eligible episodes')
        rng = np.random.default_rng(seed + env_index)
        chosen = [int(eligible[index]) for index in rng.choice(
            len(eligible), size=n_episodes, replace=False
        )]
        clips = []
        for episode in chosen:
            first, length = rows[episode]
            starts = np.unique(np.round(np.linspace(0, length - span, 4)).astype(int))
            if len(starts) != 4:
                raise ValueError(f'{env} episode {episode} has duplicate sealed starts')
            for quartile, start in enumerate(starts):
                clips.append({
                    'episode': episode,
                    'start': int(start),
                    'row': int(first + start),
                    'length': int(length),
                    'normalized_time': float(start / max(1, length - span)),
                    'quartile': quartile,
                })
        document['environments'][env] = {
            'dataset': str(path),
            'dataset_version': int(dataset.version),
            'dataset_rows': int(dataset.count_rows()),
            'episodes': chosen,
            'clips': clips,
        }
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-per-env', type=int, default=64)
    parser.add_argument('--seed', type=int, default=93072)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    source = json.loads(MANIFEST.read_text())
    document = build_sealed(source, args.episodes_per_env, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(document, handle, indent=2)
        handle.write('\n')
    print(json.dumps({
        'output': str(output),
        'sha256': sha256_file(output),
        'episodes_per_env': args.episodes_per_env,
        'clips_per_env': args.episodes_per_env * 4,
        'not_for_hyperparameter_selection': True,
    }, indent=2))


if __name__ == '__main__':
    main()
