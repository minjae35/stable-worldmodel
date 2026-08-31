"""Canonical TwoRoom collection wrapper.

Runs the unmodified baseline collector
``scripts/data/collect_tworooms.py`` into an isolated staging cache,
validates the result, then promotes it to the Design v0 canonical path
with a same-filesystem rename. Existing canonical destinations fail
fast; append and overwrite are refused.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.data.design_v0.manifest import (  # noqa: E402
    build_provenance,
    build_split_manifest,
    dataset_stats,
    indices_sha256,
    sha256_artifact,
    write_json,
)
from scripts.data.design_v0.spec import (  # noqa: E402
    COLLECT_SEED,
    COLLECTOR_TWOROOM_RELATIVE,
    CONFIG_SEED,
    TWOROOM_ACTION_DIM,
    TWOROOM_CANONICAL_NAME,
    TWOROOM_EXPECTED_EPISODES,
    TWOROOM_MAX_EPISODE_STEPS,
    assert_destination_absent,
    cache_root,
    canonical_artifacts,
    collector_default_artifacts,
)

COLLECTOR_SCRIPT = (
    _REPO_ROOT / 'scripts' / 'data' / 'collect_tworooms.py'
)
TWOROOM_SOURCE_SCRIPT = 'scripts/data/collect_tworooms.py'
TWOROOM_WRAPPER_SCRIPT = 'scripts/data/design_v0/collect_tworoom.py'

Collector = Callable[[Path], None]


def derived_collect_seed(config_seed: int = CONFIG_SEED) -> int:
    return int(np.random.default_rng(config_seed).integers(0, 1_000_000))


def verify_seeds(
    *,
    config_seed: int = CONFIG_SEED,
    collect_seed: int = COLLECT_SEED,
) -> None:
    derived = derived_collect_seed(config_seed)
    if int(config_seed) != int(CONFIG_SEED):
        raise ValueError(
            f'config_seed mismatch: got {config_seed}, '
            f'expected {CONFIG_SEED}'
        )
    if int(derived) != int(collect_seed) or int(derived) != int(
        COLLECT_SEED
    ):
        raise ValueError(
            f'collect_seed mismatch: derived {derived} from '
            f'config_seed={config_seed}, expected {COLLECT_SEED}'
        )


def collector_command(
    staging_cache: str | Path,
    *,
    num_traj: int = TWOROOM_EXPECTED_EPISODES,
    seed: int = CONFIG_SEED,
) -> list[str]:
    staging = Path(staging_cache)
    hydra_dir = staging / 'hydra'
    return [
        sys.executable,
        str(COLLECTOR_SCRIPT),
        f'cache_dir={staging}',
        f'seed={int(seed)}',
        f'num_traj={int(num_traj)}',
        f'hydra.sweep.dir={hydra_dir}',
        'hydra.job.chdir=False',
    ]


def run_baseline_collector(staging_cache: Path, num_traj: int) -> None:
    command = collector_command(staging_cache, num_traj=num_traj)
    env = os.environ.copy()
    env['STABLEWM_HOME'] = str(staging_cache)
    subprocess.run(
        command,
        check=True,
        cwd=_REPO_ROOT,
        env=env,
    )


def staging_artifact_path(staging_cache: str | Path) -> Path:
    return Path(staging_cache) / COLLECTOR_TWOROOM_RELATIVE


def make_staging_cache(
    dest: Path,
    cache_dir: str | Path | None = None,
) -> Path:
    dest = Path(dest)
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix='tworoom_staging_', dir=str(parent))
    )
    if staging.resolve() == dest.resolve():
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f'staging cache {staging} collides with canonical dest {dest}'
        )
    if staging_artifact_path(staging).resolve() == dest.resolve():
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f'staging artifact collides with canonical dest {dest}'
        )
    baseline = collector_default_artifacts(cache_dir)[0]
    if staging_artifact_path(staging).resolve() == Path(baseline).resolve():
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f'staging artifact collides with collector path {baseline}'
        )
    return staging


def promote_artifact(source: Path, dest: Path) -> Path:
    dest = assert_destination_absent(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(source, dest)
    except OSError as exc:
        raise RuntimeError(
            'failed same-filesystem rename from '
            f'{source} to {dest}: {exc}'
        ) from exc
    return dest


def validate_tworoom_artifact(
    path: str | Path,
    *,
    expected_episodes: int = TWOROOM_EXPECTED_EPISODES,
    expected_action_dim: int = TWOROOM_ACTION_DIM,
) -> dict[str, Any]:
    from stable_worldmodel.data import load_dataset

    dataset = load_dataset(str(path))
    columns = list(dataset.column_names)
    num_episodes = int(len(dataset.lengths))
    if num_episodes != int(expected_episodes):
        raise ValueError(
            f'TwoRoom episode count mismatch: got {num_episodes}, '
            f'expected {expected_episodes}'
        )
    if 'pixels' not in columns:
        raise ValueError(
            f'TwoRoom artifact is missing pixels; columns={columns}'
        )
    if 'action' not in columns:
        raise ValueError(
            f'TwoRoom artifact is missing action; columns={columns}'
        )
    if not dataset.lengths.size:
        raise ValueError('TwoRoom artifact has no episode lengths')

    action = dataset.load_episode(0)['action']
    if hasattr(action, 'detach'):
        action = action.detach().cpu().numpy()
    action_arr = np.asarray(action)
    action_shape = [int(dim) for dim in action_arr.shape]
    if action_arr.ndim < 1 or int(action_arr.shape[-1]) != int(
        expected_action_dim
    ):
        raise ValueError(
            f'TwoRoom action dim mismatch: shape={action_shape}, '
            f'expected last dim {expected_action_dim}'
        )

    pixels = dataset.load_episode(0)['pixels']
    if hasattr(pixels, 'detach'):
        pixels = pixels.detach().cpu().numpy()
    pixel_shape = [int(dim) for dim in np.asarray(pixels).shape]

    stats = dataset_stats(dataset.lengths)
    return {
        'path': str(path),
        'num_episodes': stats['num_episodes'],
        'num_transitions': stats['num_transitions'],
        'episode_length_min': stats['episode_length_min'],
        'episode_length_max': stats['episode_length_max'],
        'episode_length_mean': stats['episode_length_mean'],
        'columns': columns,
        'pixel_shape': pixel_shape,
        'action_shape': action_shape,
        'action_dim': int(action_arr.shape[-1]),
        'config_seed': CONFIG_SEED,
        'collect_seed': COLLECT_SEED,
        'max_episode_steps': TWOROOM_MAX_EPISODE_STEPS,
    }


def write_tworoom_manifests(
    dest: Path,
    inspect: dict[str, Any],
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = canonical_artifacts(cache_dir)
    artifact_sha256 = sha256_artifact(dest)
    split_payload = build_split_manifest(
        environment='TwoRoom',
        artifact_path=dest,
        num_episodes=int(inspect['num_episodes']),
        artifact_sha256=artifact_sha256,
    )
    train = split_payload['train_episode_indices']
    val = split_payload['val_episode_indices']
    if len(train) + len(val) != int(inspect['num_episodes']):
        raise ValueError('split does not cover every episode')
    if indices_sha256(train) != split_payload['train_indices_sha256']:
        raise ValueError('train split checksum mismatch')
    if indices_sha256(val) != split_payload['val_indices_sha256']:
        raise ValueError('val split checksum mismatch')

    split_path = write_json(
        artifacts['splits'] / 'tworoom.json',
        split_payload,
    )
    provenance = build_provenance(
        environment='TwoRoom',
        source={
            'script': TWOROOM_SOURCE_SCRIPT,
            'wrapper': TWOROOM_WRAPPER_SCRIPT,
            'collector_command': collector_command(
                '<staging-cache>',
                num_traj=int(inspect['num_episodes']),
            ),
        },
        artifacts={
            'canonical': {
                'path': str(dest),
                'sha256': artifact_sha256,
                'name': TWOROOM_CANONICAL_NAME,
            }
        },
        stats={
            'num_episodes': inspect['num_episodes'],
            'num_transitions': inspect['num_transitions'],
            'episode_length_min': inspect['episode_length_min'],
            'episode_length_max': inspect['episode_length_max'],
            'episode_length_mean': inspect['episode_length_mean'],
            'pixel_shape': inspect['pixel_shape'],
            'action_shape': inspect['action_shape'],
        },
        split={
            'manifest': str(split_path),
            'split_seed': split_payload['split_seed'],
            'train_episodes': len(train),
            'val_episodes': len(val),
            'train_indices_sha256': split_payload['train_indices_sha256'],
            'val_indices_sha256': split_payload['val_indices_sha256'],
        },
        schema=inspect['columns'],
        notes=(
            'Collected via unmodified scripts/data/collect_tworooms.py '
            'into isolated staging, then renamed to the canonical path.'
        ),
    )
    provenance_path = write_json(
        artifacts['manifests'] / 'tworoom.json',
        provenance,
    )
    return {
        'artifact_sha256': artifact_sha256,
        'provenance_path': str(provenance_path),
        'split_path': str(split_path),
        'train_episodes': len(train),
        'val_episodes': len(val),
        'train_indices_sha256': split_payload['train_indices_sha256'],
        'val_indices_sha256': split_payload['val_indices_sha256'],
    }


def collect_tworoom(
    *,
    cache_dir: str | Path | None = None,
    dest: str | Path | None = None,
    expected_episodes: int = TWOROOM_EXPECTED_EPISODES,
    collector: Collector | None = None,
    write_manifests: bool = True,
) -> dict[str, Any]:
    """Collect TwoRoom into staging, validate, then promote once."""
    verify_seeds()
    artifacts = canonical_artifacts(cache_dir)
    dest_path = Path(dest) if dest is not None else artifacts['tworoom']
    dest_path = assert_destination_absent(dest_path)

    cache = cache_root(cache_dir)
    baseline = cache / COLLECTOR_TWOROOM_RELATIVE
    baseline_existed = baseline.exists()

    staging = make_staging_cache(dest_path, cache_dir=cache_dir)
    staging_artifact = staging_artifact_path(staging)
    try:
        if collector is None:
            run_baseline_collector(staging, int(expected_episodes))
        else:
            collector(staging)

        if dest_path.exists():
            raise RuntimeError(
                f'collector wrote canonical dest before promotion: '
                f'{dest_path}'
            )
        if baseline.exists() and not baseline_existed:
            raise RuntimeError(
                f'collector polluted baseline path {baseline}'
            )
        if not staging_artifact.exists():
            raise FileNotFoundError(
                f'collector did not write {staging_artifact}'
            )

        inspect = validate_tworoom_artifact(
            staging_artifact,
            expected_episodes=expected_episodes,
        )
        dest_path = assert_destination_absent(dest_path)
        promote_artifact(staging_artifact, dest_path)
        inspect = validate_tworoom_artifact(
            dest_path,
            expected_episodes=expected_episodes,
        )
        report: dict[str, Any] = {
            'environment': 'TwoRoom',
            'dest': str(dest_path),
            'staging': str(staging),
            'executed': True,
            'config_seed': CONFIG_SEED,
            'collect_seed': COLLECT_SEED,
            'inspect': inspect,
        }
        if write_manifests:
            report.update(
                write_tworoom_manifests(
                    dest_path,
                    inspect,
                    cache_dir=cache_dir,
                )
            )
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Override STABLEWM_HOME when resolving the canonical path.',
    )
    parser.add_argument(
        '--dest',
        type=Path,
        default=None,
        help='Override canonical TwoRoom destination.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = collect_tworoom(
        cache_dir=args.cache_dir,
        dest=args.dest,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
