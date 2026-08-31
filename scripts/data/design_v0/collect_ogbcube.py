"""Canonical OGBCube collection wrapper.

Runs the unmodified baseline collector
``scripts/data/collect_cube.py`` into an isolated staging cache,
validates the raw multiview table, derives the front-view canonical
table with ``derive_ogbcube``, then promotes both artifacts with
same-filesystem renames. Existing canonical destinations fail fast;
append and overwrite are refused. A failed collect, validation, or
derive leaves both canonical destinations absent.
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

from scripts.data.design_v0.collect_tworoom import (  # noqa: E402
    promote_artifact,
    verify_seeds,
)
from scripts.data.design_v0.derive_ogbcube import derive_ogbcube  # noqa: E402
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
    COLLECTOR_OGBCUBE_RELATIVE,
    CONFIG_SEED,
    OGBCUBE_ACTION_DIM,
    OGBCUBE_CANONICAL_RELATIVE,
    OGBCUBE_EXPECTED_EPISODES,
    OGBCUBE_FRONT_COLUMN,
    OGBCUBE_MAX_EPISODE_STEPS,
    OGBCUBE_RAW_RELATIVE,
    assert_destination_absent,
    cache_root,
    canonical_artifacts,
    collector_default_artifacts,
)

COLLECTOR_SCRIPT = _REPO_ROOT / 'scripts' / 'data' / 'collect_cube.py'
OGBCUBE_SOURCE_SCRIPT = 'scripts/data/collect_cube.py'
OGBCUBE_WRAPPER_SCRIPT = 'scripts/data/design_v0/collect_ogbcube.py'

Collector = Callable[[Path], None]
Derive = Callable[..., dict[str, Any]]


def collector_command(
    staging_cache: str | Path,
    *,
    num_traj: int = OGBCUBE_EXPECTED_EPISODES,
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


def staging_raw_path(staging_cache: str | Path) -> Path:
    return Path(staging_cache) / COLLECTOR_OGBCUBE_RELATIVE


def staging_derived_path(staging_cache: str | Path) -> Path:
    return Path(staging_cache) / OGBCUBE_CANONICAL_RELATIVE.name


def make_staging_cache(
    dest_dir: Path,
    *,
    raw_dest: Path,
    derived_dest: Path,
    cache_dir: str | Path | None = None,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix='ogbcube_staging_', dir=str(dest_dir))
    )
    raw_staging = staging_raw_path(staging)
    derived_staging = staging_derived_path(staging)
    collisions = (
        staging.resolve() == raw_dest.resolve(),
        staging.resolve() == derived_dest.resolve(),
        raw_staging.resolve() == raw_dest.resolve(),
        derived_staging.resolve() == derived_dest.resolve(),
        raw_staging.resolve()
        == Path(collector_default_artifacts(cache_dir)[1]).resolve(),
    )
    if any(collisions):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f'staging cache {staging} collides with a protected path'
        )
    return staging


def _action_dim(action: Any) -> tuple[list[int], int]:
    if hasattr(action, 'detach'):
        action = action.detach().cpu().numpy()
    action_arr = np.asarray(action)
    shape = [int(dim) for dim in action_arr.shape]
    if action_arr.ndim < 1:
        raise ValueError(f'OGBCube action has no dims: shape={shape}')
    return shape, int(action_arr.shape[-1])


def validate_ogbcube_raw(
    path: str | Path,
    *,
    expected_episodes: int = OGBCUBE_EXPECTED_EPISODES,
    expected_action_dim: int = OGBCUBE_ACTION_DIM,
    front_column: str = OGBCUBE_FRONT_COLUMN,
) -> dict[str, Any]:
    from stable_worldmodel.data import load_dataset

    dataset = load_dataset(str(path))
    columns = list(dataset.column_names)
    num_episodes = int(len(dataset.lengths))
    if num_episodes != int(expected_episodes):
        raise ValueError(
            f'OGBCube raw episode count mismatch: got {num_episodes}, '
            f'expected {expected_episodes}'
        )
    if front_column not in columns:
        raise ValueError(
            f'OGBCube raw is missing {front_column!r}; columns={columns}'
        )
    if 'action' not in columns:
        raise ValueError(
            f'OGBCube raw is missing action; columns={columns}'
        )
    action_shape, action_dim = _action_dim(
        dataset.load_episode(0)['action']
    )
    if action_dim != int(expected_action_dim):
        raise ValueError(
            f'OGBCube action dim mismatch: shape={action_shape}, '
            f'expected last dim {expected_action_dim}'
        )
    front = dataset.load_episode(0)[front_column]
    if hasattr(front, 'detach'):
        front = front.detach().cpu().numpy()
    stats = dataset_stats(dataset.lengths)
    return {
        'path': str(path),
        'num_episodes': stats['num_episodes'],
        'num_transitions': stats['num_transitions'],
        'episode_length_min': stats['episode_length_min'],
        'episode_length_max': stats['episode_length_max'],
        'episode_length_mean': stats['episode_length_mean'],
        'columns': columns,
        'front_column': front_column,
        'pixel_shape': [int(dim) for dim in np.asarray(front).shape],
        'action_shape': action_shape,
        'action_dim': action_dim,
        'config_seed': CONFIG_SEED,
        'collect_seed': COLLECT_SEED,
        'max_episode_steps': OGBCUBE_MAX_EPISODE_STEPS,
    }


def validate_ogbcube_derived(
    path: str | Path,
    *,
    expected_episodes: int = OGBCUBE_EXPECTED_EPISODES,
    expected_action_dim: int = OGBCUBE_ACTION_DIM,
) -> dict[str, Any]:
    from stable_worldmodel.data import load_dataset

    dataset = load_dataset(str(path))
    columns = list(dataset.column_names)
    if sorted(columns) != ['action', 'pixels']:
        raise ValueError(
            f'derived schema must be pixels+action only, got {columns}'
        )
    num_episodes = int(len(dataset.lengths))
    if num_episodes != int(expected_episodes):
        raise ValueError(
            f'OGBCube derived episode count mismatch: got {num_episodes}, '
            f'expected {expected_episodes}'
        )
    action_shape, action_dim = _action_dim(
        dataset.load_episode(0)['action']
    )
    if action_dim != int(expected_action_dim):
        raise ValueError(
            f'OGBCube derived action dim mismatch: shape={action_shape}, '
            f'expected last dim {expected_action_dim}'
        )
    pixels = dataset.load_episode(0)['pixels']
    if hasattr(pixels, 'detach'):
        pixels = pixels.detach().cpu().numpy()
    stats = dataset_stats(dataset.lengths)
    return {
        'path': str(path),
        'num_episodes': stats['num_episodes'],
        'num_transitions': stats['num_transitions'],
        'columns': columns,
        'pixel_shape': [int(dim) for dim in np.asarray(pixels).shape],
        'action_shape': action_shape,
        'action_dim': action_dim,
    }


def write_ogbcube_manifests(
    *,
    raw_dest: Path,
    derived_dest: Path,
    raw_inspect: dict[str, Any],
    derived_inspect: dict[str, Any],
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = canonical_artifacts(cache_dir)
    raw_sha256 = sha256_artifact(raw_dest)
    derived_sha256 = sha256_artifact(derived_dest)
    split_payload = build_split_manifest(
        environment='OGBCube',
        artifact_path=derived_dest,
        num_episodes=int(derived_inspect['num_episodes']),
        artifact_sha256=derived_sha256,
    )
    train = split_payload['train_episode_indices']
    val = split_payload['val_episode_indices']
    if len(train) + len(val) != int(derived_inspect['num_episodes']):
        raise ValueError('split does not cover every episode')
    if indices_sha256(train) != split_payload['train_indices_sha256']:
        raise ValueError('train split checksum mismatch')
    if indices_sha256(val) != split_payload['val_indices_sha256']:
        raise ValueError('val split checksum mismatch')

    split_path = write_json(
        artifacts['splits'] / 'ogbcube.json',
        split_payload,
    )
    provenance = build_provenance(
        environment='OGBCube',
        source={
            'script': OGBCUBE_SOURCE_SCRIPT,
            'wrapper': OGBCUBE_WRAPPER_SCRIPT,
            'derive': 'scripts/data/design_v0/derive_ogbcube.py',
            'front_column': OGBCUBE_FRONT_COLUMN,
            'collector_command': collector_command(
                '<staging-cache>',
                num_traj=int(raw_inspect['num_episodes']),
            ),
        },
        artifacts={
            'raw': {
                'path': str(raw_dest),
                'sha256': raw_sha256,
                'name': OGBCUBE_RAW_RELATIVE.as_posix(),
            },
            'canonical': {
                'path': str(derived_dest),
                'sha256': derived_sha256,
                'name': OGBCUBE_CANONICAL_RELATIVE.as_posix(),
            },
        },
        stats={
            'raw': {
                'num_episodes': raw_inspect['num_episodes'],
                'num_transitions': raw_inspect['num_transitions'],
                'pixel_shape': raw_inspect['pixel_shape'],
                'action_shape': raw_inspect['action_shape'],
            },
            'derived': {
                'num_episodes': derived_inspect['num_episodes'],
                'num_transitions': derived_inspect['num_transitions'],
                'pixel_shape': derived_inspect['pixel_shape'],
                'action_shape': derived_inspect['action_shape'],
            },
        },
        split={
            'manifest': str(split_path),
            'split_seed': split_payload['split_seed'],
            'train_episodes': len(train),
            'val_episodes': len(val),
            'train_indices_sha256': split_payload['train_indices_sha256'],
            'val_indices_sha256': split_payload['val_indices_sha256'],
        },
        schema=derived_inspect['columns'],
        notes=(
            'Collected via unmodified scripts/data/collect_cube.py into '
            'isolated staging, derived pixels_front_pixels→pixels, then '
            'renamed both artifacts to canonical paths. Raw multiview is '
            'preserved.'
        ),
    )
    provenance_path = write_json(
        artifacts['manifests'] / 'ogbcube.json',
        provenance,
    )
    return {
        'raw_sha256': raw_sha256,
        'derived_sha256': derived_sha256,
        'provenance_path': str(provenance_path),
        'split_path': str(split_path),
        'train_episodes': len(train),
        'val_episodes': len(val),
        'train_indices_sha256': split_payload['train_indices_sha256'],
        'val_indices_sha256': split_payload['val_indices_sha256'],
    }


def collect_ogbcube(
    *,
    cache_dir: str | Path | None = None,
    expected_episodes: int = OGBCUBE_EXPECTED_EPISODES,
    front_column: str = OGBCUBE_FRONT_COLUMN,
    collector: Collector | None = None,
    derive: Derive | None = None,
    write_manifests: bool = True,
) -> dict[str, Any]:
    """Collect raw OGBCube, derive front-view, then promote both once."""
    verify_seeds()
    artifacts = canonical_artifacts(cache_dir)
    raw_dest = assert_destination_absent(artifacts['ogbcube_raw'])
    derived_dest = assert_destination_absent(artifacts['ogbcube'])

    cache = cache_root(cache_dir)
    baseline = cache / COLLECTOR_OGBCUBE_RELATIVE
    baseline_existed = baseline.exists()

    staging = make_staging_cache(
        raw_dest.parent,
        raw_dest=raw_dest,
        derived_dest=derived_dest,
        cache_dir=cache_dir,
    )
    raw_staging = staging_raw_path(staging)
    derived_staging = staging_derived_path(staging)
    promoted_raw = False
    promoted_derived = False
    try:
        if collector is None:
            run_baseline_collector(staging, int(expected_episodes))
        else:
            collector(staging)

        if raw_dest.exists() or derived_dest.exists():
            raise RuntimeError(
                'collector wrote a canonical dest before promotion'
            )
        if baseline.exists() and not baseline_existed:
            raise RuntimeError(
                f'collector polluted baseline path {baseline}'
            )
        if not raw_staging.exists():
            raise FileNotFoundError(
                f'collector did not write {raw_staging}'
            )

        raw_inspect = validate_ogbcube_raw(
            raw_staging,
            expected_episodes=expected_episodes,
            front_column=front_column,
        )
        if derive is None:
            derive_report = derive_ogbcube(
                raw_staging,
                derived_staging,
                front_column=front_column,
            )
        else:
            derive_report = derive(
                raw_staging,
                derived_staging,
                front_column=front_column,
            )
        if not derived_staging.exists():
            raise FileNotFoundError(
                f'derive did not write {derived_staging}'
            )
        derived_inspect = validate_ogbcube_derived(
            derived_staging,
            expected_episodes=expected_episodes,
        )
        raw_dest = assert_destination_absent(raw_dest)
        derived_dest = assert_destination_absent(derived_dest)
        promote_artifact(raw_staging, raw_dest)
        promoted_raw = True
        promote_artifact(derived_staging, derived_dest)
        promoted_derived = True

        raw_inspect = validate_ogbcube_raw(
            raw_dest,
            expected_episodes=expected_episodes,
            front_column=front_column,
        )
        derived_inspect = validate_ogbcube_derived(
            derived_dest,
            expected_episodes=expected_episodes,
        )
        report: dict[str, Any] = {
            'environment': 'OGBCube',
            'raw_dest': str(raw_dest),
            'derived_dest': str(derived_dest),
            'staging': str(staging),
            'executed': True,
            'config_seed': CONFIG_SEED,
            'collect_seed': COLLECT_SEED,
            'front_column': front_column,
            'raw_inspect': raw_inspect,
            'derived_inspect': derived_inspect,
            'derive': derive_report,
        }
        if write_manifests:
            report.update(
                write_ogbcube_manifests(
                    raw_dest=raw_dest,
                    derived_dest=derived_dest,
                    raw_inspect=raw_inspect,
                    derived_inspect=derived_inspect,
                    cache_dir=cache_dir,
                )
            )
        return report
    except Exception:
        if promoted_derived and derived_dest.exists():
            shutil.rmtree(derived_dest, ignore_errors=True)
        if promoted_raw and raw_dest.exists():
            shutil.rmtree(raw_dest, ignore_errors=True)
        raise
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
        '--front-column',
        default=OGBCUBE_FRONT_COLUMN,
        help='Raw front-view column renamed to pixels.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = collect_ogbcube(
        cache_dir=args.cache_dir,
        front_column=args.front_column,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
