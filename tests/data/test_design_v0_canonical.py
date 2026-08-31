"""Fixture tests for Design v0 canonical dataset acquisition tooling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data.design_v0.acquire_pusht import (
    acquire_pusht,
    build_acquire_plan,
    decompress_zst,
    hf_download_command,
    hf_download_kwargs,
    parse_hf_tree_entry,
    verify_file_sha256,
    verify_h5_counts,
    verify_lfs_sha256,
    verify_pusht_counts,
    verify_zst_size,
)
from scripts.data.design_v0.derive_ogbcube import (
    derive_ogbcube,
    resolve_front_column,
)
from scripts.data.design_v0.dryrun_collect import dryrun_dest, run_dryrun
from scripts.data.design_v0.manifest import (
    build_provenance,
    build_split_manifest,
    episode_split,
    git_commit,
    indices_sha256,
    sha256_artifact,
    sha256_file,
    sha256_tree,
    write_json,
)
from scripts.data.design_v0.spec import (
    COLLECT_SEED,
    CONFIG_SEED,
    PUSHT_EXPECTED_EPISODES,
    PUSHT_EXPECTED_TRANSITIONS,
    PUSHT_HF_FILENAME,
    PUSHT_HF_REVISION,
    PUSHT_LFS_SHA256,
    SPLIT_SEED,
    TRAIN_FRACTION,
    assert_destination_absent,
    canonical_root,
)


def test_collect_seed_matches_collector_rng():
    derived = int(
        np.random.default_rng(CONFIG_SEED).integers(0, 1_000_000)
    )
    assert COLLECT_SEED == derived == 724895


def test_episode_split_is_90_10_and_covers_all_ids():
    train, val = episode_split(10)
    assert len(train) == 9
    assert len(val) == 1
    assert sorted(train + val) == list(range(10))
    assert set(train).isdisjoint(val)


def test_episode_split_is_deterministic():
    first = episode_split(2000, split_seed=SPLIT_SEED)
    second = episode_split(2000, split_seed=SPLIT_SEED)
    assert first == second
    other = episode_split(2000, split_seed=SPLIT_SEED + 1)
    assert first != other


def test_pusht_episode_split_sizes():
    train, val = episode_split(PUSHT_EXPECTED_EPISODES)
    train_n = int(PUSHT_EXPECTED_EPISODES * TRAIN_FRACTION)
    val_n = PUSHT_EXPECTED_EPISODES - train_n
    assert len(train) == train_n == 16816
    assert len(val) == val_n == 1869
    assert TRAIN_FRACTION == 0.9
    assert sorted(train + val) == list(range(PUSHT_EXPECTED_EPISODES))


def test_episode_split_rejects_empty_side():
    with pytest.raises(ValueError, match='non-empty'):
        episode_split(1)


def test_artifact_digest_file_and_tree(tmp_path):
    file_path = tmp_path / 'blob.bin'
    file_path.write_bytes(b'design-v0')
    digest = sha256_file(file_path)
    assert digest == sha256_artifact(file_path)
    assert len(digest) == 64

    tree = tmp_path / 'tree'
    (tree / 'a').mkdir(parents=True)
    (tree / 'a' / 'one.txt').write_text('one')
    (tree / 'two.txt').write_text('two')
    tree_digest = sha256_tree(tree)
    assert tree_digest == sha256_artifact(tree)
    assert tree_digest != digest

    (tree / 'two.txt').write_text('changed')
    assert sha256_tree(tree) != tree_digest


def test_provenance_records_both_seeds_and_git(tmp_path):
    split = build_split_manifest(
        environment='TwoRoom',
        artifact_path='tworoom_expert.lance',
        num_episodes=10,
        artifact_sha256='abc',
    )
    payload = build_provenance(
        environment='TwoRoom',
        source={'script': 'scripts/data/collect_tworooms.py'},
        artifacts={'canonical': {'path': 'tworoom_expert.lance'}},
        stats={'num_episodes': 10, 'num_transitions': 100},
        split={
            'manifest': 'splits/tworoom.json',
            'split_seed': SPLIT_SEED,
        },
        schema=['pixels', 'action'],
        git_sha=git_commit(),
    )
    assert payload['config_seed'] == 3072
    assert payload['collect_seed'] == 724895
    assert payload['source']['script'] == (
        'scripts/data/collect_tworooms.py'
    )
    written = write_json(tmp_path / 'tworoom.json', payload)
    loaded = json.loads(written.read_text())
    assert loaded['config_seed'] == CONFIG_SEED
    split_path = write_json(tmp_path / 'split.json', split)
    split_loaded = json.loads(split_path.read_text())
    assert split_loaded['train_indices_sha256'] == indices_sha256(
        split_loaded['train_episode_indices']
    )
    assert split_loaded['unit'] == 'episode'


def test_pusht_plan_pins_revision_and_does_not_download():
    def boom(**kwargs):
        raise AssertionError(f'unexpected download {kwargs}')

    plan = acquire_pusht(download=False, downloader=boom)
    kwargs = hf_download_kwargs()
    assert kwargs['revision'] == PUSHT_HF_REVISION
    assert kwargs['filename'] == PUSHT_HF_FILENAME
    assert kwargs['repo_id'] == 'quentinll/lewm-pusht'
    command = hf_download_command()
    assert PUSHT_HF_REVISION in command
    assert '--revision' in command
    assert plan['executed'] is False
    assert plan['revision'] == PUSHT_HF_REVISION
    assert plan['expected_lfs_sha256'] == PUSHT_LFS_SHA256
    assert plan['expected_episodes'] == PUSHT_EXPECTED_EPISODES
    assert plan['expected_transitions'] == PUSHT_EXPECTED_TRANSITIONS


def test_pusht_tree_metadata_and_lfs_checksum():
    tree = [
        {'path': 'README.md', 'size': 136},
        {
            'path': PUSHT_HF_FILENAME,
            'size': 13_136_247_974,
            'lfs': {'oid': PUSHT_LFS_SHA256},
        },
    ]
    entry = parse_hf_tree_entry(tree)
    verify_zst_size(entry['size'])
    verify_lfs_sha256(entry['lfs']['oid'])
    with pytest.raises(ValueError, match='LFS SHA-256'):
        verify_lfs_sha256('0' * 64)


def test_pusht_count_and_h5_stats(tmp_path):
    verify_pusht_counts(
        PUSHT_EXPECTED_EPISODES,
        PUSHT_EXPECTED_TRANSITIONS,
    )
    with pytest.raises(ValueError, match='episode count'):
        verify_pusht_counts(1999, PUSHT_EXPECTED_TRANSITIONS)
    with pytest.raises(ValueError, match='transition count'):
        verify_pusht_counts(PUSHT_EXPECTED_EPISODES, 1)

    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'tiny.h5'
    with h5py.File(path, 'w') as handle:
        handle.create_dataset(
            'ep_len', data=np.array([10, 20], dtype=np.int32)
        )
        handle.create_dataset(
            'ep_offset', data=np.array([0, 10], dtype=np.int32)
        )
    with pytest.raises(ValueError, match='episode count'):
        verify_h5_counts(path)

    blob = tmp_path / 'src.bin'
    blob.write_bytes(b'abc')
    digest = sha256_file(blob)
    verify_file_sha256(blob, digest)
    with pytest.raises(ValueError, match='SHA-256'):
        verify_file_sha256(blob, '0' * 64)


def test_acquire_download_hook_verifies_source_sha(tmp_path):
    dest_dir = tmp_path / 'pusht'
    zst = dest_dir / PUSHT_HF_FILENAME
    h5 = dest_dir / 'pusht_expert_train.h5'

    def fake_download(**kwargs):
        assert kwargs['revision'] == PUSHT_HF_REVISION
        zst.parent.mkdir(parents=True, exist_ok=True)
        zst.write_bytes(b'zst-bytes')
        return str(zst)

    def fake_run(command, check):
        assert command[0] == 'zstd'
        h5.write_bytes(b'not-a-real-h5')

    # Source SHA will not match the official LFS oid.
    with pytest.raises(ValueError, match='LFS SHA-256'):
        acquire_pusht(
            dest_dir=dest_dir,
            download=True,
            downloader=fake_download,
            runner=fake_run,
        )


def test_decompress_refuses_existing_dest(tmp_path):
    src = tmp_path / 'in.zst'
    dest = tmp_path / 'out.h5'
    src.write_bytes(b'x')
    dest.write_bytes(b'y')

    def fake_run(command, check):
        raise AssertionError('zstd should not run')

    with pytest.raises(FileExistsError):
        decompress_zst(src, dest, runner=fake_run)


def test_dryrun_refuses_canonical_and_collector_paths(tmp_path):
    cache = tmp_path / 'swm'
    with pytest.raises(ValueError, match='canonical'):
        dryrun_dest(canonical_root(cache), 'tworoom', cache_dir=cache)
    collector = cache / 'datasets'
    with pytest.raises(ValueError, match='protected'):
        dryrun_dest(collector, 'tworoom', cache_dir=cache)


def test_dryrun_isolated_path_and_no_collect(tmp_path):
    isolated = tmp_path / 'isolated'
    isolated.mkdir()
    report = run_dryrun(
        'tworoom',
        isolated,
        cache_dir=tmp_path / 'swm',
        collect=False,
    )
    dest = Path(report['dest'])
    assert dest.parent == isolated
    assert dest.name == 'tworoom_expert.lance'
    assert not dest.exists()
    assert report['episodes'] == 3
    assert report['num_envs'] == 1
    assert report['collect_seed'] == COLLECT_SEED


def test_dryrun_refuses_existing_dest(tmp_path):
    isolated = tmp_path / 'isolated'
    isolated.mkdir()
    dest = isolated / 'tworoom_expert.lance'
    dest.mkdir()
    with pytest.raises(FileExistsError):
        dryrun_dest(isolated, 'tworoom', cache_dir=tmp_path / 'swm')


def test_existing_destination_fail_fast():
    with pytest.raises(FileExistsError):
        assert_destination_absent(Path(__file__))


def test_ogbcube_front_column_must_be_explicit():
    columns = ['pixels_front_pixels', 'pixels_side_pixels', 'action']
    assert (
        resolve_front_column(columns, 'pixels_front_pixels')
        == 'pixels_front_pixels'
    )
    with pytest.raises(ValueError, match='front-view column'):
        resolve_front_column(columns, 'pixels')


def _write_multiview_lance(path: Path) -> None:
    pytest.importorskip('lancedb')
    from stable_worldmodel.data import LanceWriter

    front = np.full((8, 8, 3), 11, dtype=np.uint8)
    side = np.full((8, 8, 3), 99, dtype=np.uint8)
    with LanceWriter(path) as writer:
        for _ in range(4):
            writer.write_episode(
                {
                    'pixels_front_pixels': [front.copy() for _ in range(3)],
                    'pixels_side_pixels': [side.copy() for _ in range(3)],
                    'action': [
                        np.arange(5, dtype=np.float32) for _ in range(3)
                    ],
                    'proprio': [
                        np.ones(4, dtype=np.float32) for _ in range(3)
                    ],
                }
            )


def test_ogbcube_column_rename_keeps_raw(tmp_path):
    pytest.importorskip('lancedb')
    from stable_worldmodel.data import LanceDataset

    source = tmp_path / 'raw.lance'
    dest = tmp_path / 'canonical.lance'
    _write_multiview_lance(source)
    before = sha256_artifact(source)

    report = derive_ogbcube(
        source,
        dest,
        front_column='pixels_front_pixels',
    )
    assert report['num_episodes'] == 4
    assert sorted(report['dest_columns']) == ['action', 'pixels']
    assert sha256_artifact(source) == before

    derived = LanceDataset(path=dest)
    sample = derived.load_episode(0)
    pixels = sample['pixels']
    if hasattr(pixels, 'detach'):
        pixels = pixels.detach().cpu().numpy()
    assert int(np.asarray(pixels).max()) == 11
    assert int(np.asarray(pixels).min()) == 11
    assert 'pixels_side_pixels' not in derived.column_names
    assert 'proprio' not in derived.column_names


def test_ogbcube_existing_dest_does_not_append(tmp_path):
    pytest.importorskip('lancedb')
    from stable_worldmodel.data import LanceWriter

    source = tmp_path / 'raw.lance'
    dest = tmp_path / 'canonical.lance'
    _write_multiview_lance(source)
    dest.mkdir()
    (dest / 'marker.txt').write_text('existing')
    before = sha256_artifact(source)
    with pytest.raises(FileExistsError):
        derive_ogbcube(
            source,
            dest,
            front_column='pixels_front_pixels',
        )
    assert sha256_artifact(source) == before
    assert (dest / 'marker.txt').read_text() == 'existing'

    # Writer mode=error also fails if a lance table is already present.
    existing = tmp_path / 'already.lance'
    with LanceWriter(existing) as writer:
        writer.write_episode(
            {
                'pixels': [
                    np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)
                ],
                'action': [np.zeros(5, dtype=np.float32) for _ in range(2)],
            }
        )
    with pytest.raises(FileExistsError):
        derive_ogbcube(
            source,
            existing,
            front_column='pixels_front_pixels',
        )


def test_build_acquire_plan_dest_dir(tmp_path):
    plan = build_acquire_plan(dest_dir=tmp_path)
    assert plan['canonical_h5'].endswith('pusht_expert_train.h5')
    assert PUSHT_HF_REVISION in plan['hf_download_command']
