"""Fixture tests for the Design v0 OGBCube canonical collection wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data.design_v0.collect_ogbcube import (
    COLLECTOR_SCRIPT,
    collect_ogbcube,
    collector_command,
    staging_derived_path,
    staging_raw_path,
    validate_ogbcube_derived,
    validate_ogbcube_raw,
    verify_seeds,
)
from scripts.data.design_v0.manifest import (
    episode_split,
    indices_sha256,
    sha256_artifact,
)
from scripts.data.design_v0.spec import (
    COLLECT_SEED,
    COLLECTOR_OGBCUBE_RELATIVE,
    CONFIG_SEED,
    OGBCUBE_ACTION_DIM,
    OGBCUBE_CANONICAL_RELATIVE,
    OGBCUBE_EXPECTED_EPISODES,
    OGBCUBE_FRONT_COLUMN,
    OGBCUBE_RAW_RELATIVE,
    canonical_artifacts,
    collector_default_artifacts,
)


N_EPISODES = 10
STEPS = 3


def _write_ogbcube_raw_lance(
    path: Path,
    *,
    num_episodes: int = N_EPISODES,
    action_dim: int = OGBCUBE_ACTION_DIM,
    include_front: bool = True,
) -> None:
    pytest.importorskip('lancedb')
    from stable_worldmodel.data import LanceWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    episode: dict[str, list] = {
        'pixels_side_pixels': [
            np.full((8, 8, 3), 99, dtype=np.uint8) for _ in range(STEPS)
        ],
        'action': [
            np.arange(action_dim, dtype=np.float32) for _ in range(STEPS)
        ],
        'proprio': [np.ones(4, dtype=np.float32) for _ in range(STEPS)],
    }
    if include_front:
        episode[OGBCUBE_FRONT_COLUMN] = [
            np.full((8, 8, 3), 11, dtype=np.uint8) for _ in range(STEPS)
        ]
    with LanceWriter(path) as writer:
        for _ in range(num_episodes):
            writer.write_episode(
                {
                    key: [item.copy() for item in values]
                    for key, values in episode.items()
                }
            )


def _ok_collector(staging: Path, **kwargs) -> None:
    _write_ogbcube_raw_lance(staging_raw_path(staging), **kwargs)


def _mark(path: Path, text: str = 'existing') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    (path / 'marker.txt').write_text(text)


def test_existing_raw_dest_fails_fast(tmp_path):
    dest = canonical_artifacts(tmp_path)['ogbcube_raw']
    _mark(dest)

    def boom(staging: Path) -> None:
        raise AssertionError(f'collector should not run: {staging}')

    with pytest.raises(FileExistsError, match='already exists'):
        collect_ogbcube(
            cache_dir=tmp_path,
            collector=boom,
            expected_episodes=N_EPISODES,
        )
    assert (dest / 'marker.txt').read_text() == 'existing'
    assert not canonical_artifacts(tmp_path)['ogbcube'].exists()


def test_existing_derived_dest_fails_fast(tmp_path):
    dest = canonical_artifacts(tmp_path)['ogbcube']
    _mark(dest)

    def boom(staging: Path) -> None:
        raise AssertionError(f'collector should not run: {staging}')

    with pytest.raises(FileExistsError, match='already exists'):
        collect_ogbcube(
            cache_dir=tmp_path,
            collector=boom,
            expected_episodes=N_EPISODES,
        )
    assert (dest / 'marker.txt').read_text() == 'existing'
    assert not canonical_artifacts(tmp_path)['ogbcube_raw'].exists()


def test_staging_is_isolated_from_canonical_and_collector_paths(tmp_path):
    seen: dict[str, Path] = {}

    def collector(staging: Path) -> None:
        seen['staging'] = Path(staging)
        seen['raw'] = staging_raw_path(staging)
        seen['derived'] = staging_derived_path(staging)
        artifacts = canonical_artifacts(tmp_path)
        baseline = collector_default_artifacts(tmp_path)[1]
        assert staging.resolve() != artifacts['ogbcube_raw'].resolve()
        assert staging.resolve() != artifacts['ogbcube'].resolve()
        assert seen['raw'].resolve() != artifacts['ogbcube_raw'].resolve()
        assert (
            seen['derived'].resolve() != artifacts['ogbcube'].resolve()
        )
        assert seen['raw'].resolve() != Path(baseline).resolve()
        _ok_collector(staging)

    report = collect_ogbcube(
        cache_dir=tmp_path,
        collector=collector,
        expected_episodes=N_EPISODES,
    )
    artifacts = canonical_artifacts(tmp_path)
    assert Path(report['raw_dest']) == artifacts['ogbcube_raw']
    assert Path(report['derived_dest']) == artifacts['ogbcube']
    assert Path(report['raw_dest']).exists()
    assert Path(report['derived_dest']).exists()
    assert seen['staging'].name.startswith('ogbcube_staging_')
    assert seen['staging'].parent == artifacts['ogbcube_raw'].parent
    assert not seen['staging'].exists()
    assert not seen['raw'].exists()
    assert not seen['derived'].exists()
    assert not (tmp_path / COLLECTOR_OGBCUBE_RELATIVE).exists()


def test_collector_failure_leaves_canonical_dests_absent(tmp_path):
    artifacts = canonical_artifacts(tmp_path)

    def boom(staging: Path) -> None:
        raise RuntimeError('collector exploded')

    with pytest.raises(RuntimeError, match='collector exploded'):
        collect_ogbcube(
            cache_dir=tmp_path,
            collector=boom,
            expected_episodes=N_EPISODES,
        )
    assert not artifacts['ogbcube_raw'].exists()
    assert not artifacts['ogbcube'].exists()
    dest_dir = artifacts['ogbcube_raw'].parent
    assert not list(dest_dir.glob('ogbcube_staging_*'))


def test_raw_validation_failure_does_not_promote(tmp_path):
    artifacts = canonical_artifacts(tmp_path)

    def bad_count(staging: Path) -> None:
        _ok_collector(staging, num_episodes=N_EPISODES - 1)

    with pytest.raises(ValueError, match='episode count'):
        collect_ogbcube(
            cache_dir=tmp_path,
            collector=bad_count,
            expected_episodes=N_EPISODES,
        )
    assert not artifacts['ogbcube_raw'].exists()
    assert not artifacts['ogbcube'].exists()

    def missing_front(staging: Path) -> None:
        _ok_collector(staging, include_front=False)

    with pytest.raises(ValueError, match='missing'):
        collect_ogbcube(
            cache_dir=tmp_path,
            collector=missing_front,
            expected_episodes=N_EPISODES,
        )
    assert not artifacts['ogbcube_raw'].exists()
    assert not artifacts['ogbcube'].exists()

    def wrong_action(staging: Path) -> None:
        _ok_collector(staging, action_dim=2)

    with pytest.raises(ValueError, match='action dim'):
        collect_ogbcube(
            cache_dir=tmp_path,
            collector=wrong_action,
            expected_episodes=N_EPISODES,
        )
    assert not artifacts['ogbcube_raw'].exists()
    assert not artifacts['ogbcube'].exists()
    assert not (tmp_path / COLLECTOR_OGBCUBE_RELATIVE).exists()


def test_derive_failure_does_not_promote(tmp_path):
    artifacts = canonical_artifacts(tmp_path)

    def boom_derive(source, dest, *, front_column):
        raise RuntimeError('derive exploded')

    with pytest.raises(RuntimeError, match='derive exploded'):
        collect_ogbcube(
            cache_dir=tmp_path,
            collector=_ok_collector,
            derive=boom_derive,
            expected_episodes=N_EPISODES,
        )
    assert not artifacts['ogbcube_raw'].exists()
    assert not artifacts['ogbcube'].exists()


def test_derived_validation_failure_does_not_promote(tmp_path):
    artifacts = canonical_artifacts(tmp_path)
    pytest.importorskip('lancedb')
    from stable_worldmodel.data import LanceWriter

    def bad_derive(source, dest, *, front_column):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with LanceWriter(dest) as writer:
            for _ in range(N_EPISODES):
                writer.write_episode(
                    {
                        'pixels': [
                            np.zeros((8, 8, 3), dtype=np.uint8)
                            for _ in range(STEPS)
                        ],
                        'action': [
                            np.arange(2, dtype=np.float32)
                            for _ in range(STEPS)
                        ],
                    }
                )
        return {'dest': str(dest)}

    with pytest.raises(ValueError, match='action dim'):
        collect_ogbcube(
            cache_dir=tmp_path,
            collector=_ok_collector,
            derive=bad_derive,
            expected_episodes=N_EPISODES,
        )
    assert not artifacts['ogbcube_raw'].exists()
    assert not artifacts['ogbcube'].exists()


def test_successful_staging_promotes_and_writes_manifests(tmp_path):
    report = collect_ogbcube(
        cache_dir=tmp_path,
        collector=_ok_collector,
        expected_episodes=N_EPISODES,
    )
    artifacts = canonical_artifacts(tmp_path)
    raw_dest = Path(report['raw_dest'])
    derived_dest = Path(report['derived_dest'])
    assert raw_dest.exists()
    assert derived_dest.exists()
    assert raw_dest == artifacts['ogbcube_raw']
    assert derived_dest == artifacts['ogbcube']
    assert raw_dest.name == OGBCUBE_RAW_RELATIVE.name
    assert derived_dest.name == OGBCUBE_CANONICAL_RELATIVE.name
    assert report['config_seed'] == CONFIG_SEED
    assert report['collect_seed'] == COLLECT_SEED
    assert report['front_column'] == OGBCUBE_FRONT_COLUMN
    assert report['raw_inspect']['num_episodes'] == N_EPISODES
    assert report['raw_inspect']['action_dim'] == OGBCUBE_ACTION_DIM
    assert OGBCUBE_FRONT_COLUMN in report['raw_inspect']['columns']
    assert 'pixels_side_pixels' in report['raw_inspect']['columns']
    assert report['derived_inspect']['num_episodes'] == N_EPISODES
    assert sorted(report['derived_inspect']['columns']) == [
        'action',
        'pixels',
    ]
    assert report['train_episodes'] == 9
    assert report['val_episodes'] == 1

    raw = validate_ogbcube_raw(raw_dest, expected_episodes=N_EPISODES)
    derived = validate_ogbcube_derived(
        derived_dest,
        expected_episodes=N_EPISODES,
    )
    assert raw['num_transitions'] == N_EPISODES * STEPS
    assert derived['num_transitions'] == N_EPISODES * STEPS
    assert sha256_artifact(raw_dest) == report['raw_sha256']
    assert sha256_artifact(derived_dest) == report['derived_sha256']

    provenance = json.loads(
        Path(report['provenance_path']).read_text()
    )
    split = json.loads(Path(report['split_path']).read_text())
    assert provenance['source']['script'] == (
        'scripts/data/collect_cube.py'
    )
    assert provenance['source']['front_column'] == OGBCUBE_FRONT_COLUMN
    assert provenance['config_seed'] == CONFIG_SEED
    assert provenance['collect_seed'] == COLLECT_SEED
    assert split['train_indices_sha256'] == indices_sha256(
        split['train_episode_indices']
    )
    assert split['val_indices_sha256'] == indices_sha256(
        split['val_episode_indices']
    )
    assert not (tmp_path / COLLECTOR_OGBCUBE_RELATIVE).exists()
    dest_dir = raw_dest.parent
    assert not list(dest_dir.glob('ogbcube_staging_*'))


def test_episode_split_10000_is_9000_1000():
    train, val = episode_split(OGBCUBE_EXPECTED_EPISODES)
    assert len(train) == 9000
    assert len(val) == 1000
    assert sorted(train + val) == list(
        range(OGBCUBE_EXPECTED_EPISODES)
    )
    assert set(train).isdisjoint(val)


def test_collector_command_uses_baseline_script_and_seeds():
    command = collector_command('/tmp/staging')
    assert command[1] == str(COLLECTOR_SCRIPT)
    assert COLLECTOR_SCRIPT.name == 'collect_cube.py'
    assert 'cache_dir=/tmp/staging' in command
    assert f'seed={CONFIG_SEED}' in command
    assert f'num_traj={OGBCUBE_EXPECTED_EPISODES}' in command
    verify_seeds()
    with pytest.raises(ValueError, match='config_seed'):
        verify_seeds(config_seed=0)
