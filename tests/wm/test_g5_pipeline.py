import json
from pathlib import Path

import pytest
import torch

from scripts.train.launch_g5_object_dynamics import command_for
from stable_worldmodel.wm.spatial.g2 import config_sha256
from stable_worldmodel.wm.spatial.g5 import (
    FINAL_STEP, NEW_UPDATES, SOURCE_STEP, run_config, validate_h1_source,
)
from stable_worldmodel.wm.spatial.g5_objective import G5_ARMS


def _source_payload():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    for state in optimizer.state.values():
        state['step'] = torch.tensor(float(SOURCE_STEP))
    config = {'history': 1, 'loss': 'mse', 'architecture_changed': False}
    return {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'step': SOURCE_STEP, 'config_sha256': config_sha256(config),
        'extra': {'rng_state': {'present': True}},
    }, config


def _plan():
    return {
        'seed': 3072, 'batch_per_environment': 16,
        'objective': {'motion_blend': .5, 'relative_cap': 4., 'margin_fraction': .05},
        'calibration': {
            arm: {'ranking_lambda': .1 if 'action' in arm or 'combined' in arm else 0.,
                  'objective_scale': 1.} for arm in G5_ARMS
        },
        'source': {'checkpoint_sha256': 'a', 'model_sha256': 'b'},
        'train_cache_key': 'c', 'schedule_set_sha256': 'd',
        'derangement_set_sha256': 'e', 'common_manifest_sha256': 'f',
        'normalizer_sha256': 'g', 'dino_sha256': 'h',
        'training_columns': ['episode_idx', 'pixels', 'action'],
        'forbidden_training_inputs': ['object_state', 'mask', 'reward', 'contact'],
    }


def test_fixed_budget_and_cumulative_steps():
    assert SOURCE_STEP == 2000 and NEW_UPDATES == 500 and FINAL_STEP == 2500
    assert len(G5_ARMS) * NEW_UPDATES == 2000


def test_all_arm_configs_differ_only_in_objective_and_output():
    configs = [run_config(_plan(), arm, Path('/tmp') / arm) for arm in G5_ARMS]
    ignored = {'arm', 'motion_balanced', 'action_sensitive', 'ranking_lambda',
               'objective_scale', 'output_dir'}
    signatures = [{k: v for k, v in config.items() if k not in ignored} for config in configs]
    assert all(value == signatures[0] for value in signatures[1:])
    assert all(c['training_columns'] == ['episode_idx', 'pixels', 'action'] for c in configs)


def test_source_validation_requires_exact_complete_resume():
    payload, config = _source_payload()
    report = validate_h1_source(payload, config)
    assert report['step'] == SOURCE_STEP and report['rng_present']
    payload['step'] = 1999
    with pytest.raises(ValueError, match='step-2000'):
        validate_h1_source(payload, config)


def test_launcher_smoke_is_prefix_of_final_run(tmp_path):
    for arm in G5_ARMS:
        smoke = command_for('/venv/stable-wm/bin/python', tmp_path, arm, False)
        resume = command_for('/venv/stable-wm/bin/python', tmp_path, arm, True)
        assert smoke[smoke.index('--stop-after-update') + 1] == '10'
        assert '--resume' in resume
        assert str(tmp_path / arm / 'checkpoint_step2010.pt') in resume


def test_plan_contains_no_training_supervision_labels():
    serialized = json.dumps(_plan()['training_columns'])
    for forbidden in ('state', 'mask', 'reward', 'contact', 'event'):
        assert forbidden not in serialized
