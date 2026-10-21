import io
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stable_worldmodel.wm.spatial.g2 import (
    ARM_SPECS,
    arm_config,
    assert_label_free_columns,
    calibrate_loss_scales,
    clip_rows,
    fairness_signature,
    load_resume_checkpoint,
    make_index_schedule,
    save_checkpoint_no_overwrite,
    schedule_sha256,
    spatial_prediction_loss,
    tensor_state_sha256,
    validate_manifest,
)
from stable_worldmodel.wm.spatial.g2_eval import (
    challenge_mask,
    probe_metric_report,
    token_error_metrics,
)
from stable_worldmodel.wm.spatial.model import SpatialDynamics


def _manifest():
    document = {
        'frameskip': 5,
        'span': 10,
        'clips_per_episode': 2,
    }
    for env in ('pusht', 'tworoom', 'cube'):
        split = {'train': [0, 1], 'val': [2], 'test': [3]}
        body = {'env': env, 'split': split}
        for sampling in ('s0', 's1'):
            body[sampling] = {}
            for split_name, episodes in split.items():
                body[sampling][split_name] = [
                    {
                        'episode': episode,
                        'start': offset,
                        'row': episode * 100 + offset,
                        'length': 20,
                        'normalized_time': offset / 10,
                        'quartile': offset,
                    }
                    for episode in episodes
                    for offset in (0, 1)
                ]
        document[env] = body
    return document


def _tiny_model():
    return SpatialDynamics(
        token_dim=8,
        hidden=16,
        depth=1,
        heads=4,
        mlp_ratio=2,
        grid=2,
        dropout=0,
    )


def test_four_arm_factorization_and_fairness_signature():
    assert {(spec.sampling, spec.loss) for spec in ARM_SPECS.values()} == {
        ('s0', 'mse'),
        ('s1', 'mse'),
        ('s0', 'l1'),
        ('s1', 'l1'),
    }
    base = {'seed': 7, 'steps': 500, 'batch': 16, 'output_dir': 'ignored'}
    signatures = [fairness_signature(arm_config(base, arm), ARM_SPECS[arm]) for arm in 'ABCD']
    assert signatures.count(signatures[0]) == 4


def test_manifest_is_episode_disjoint_and_balanced():
    summary = validate_manifest(_manifest())
    assert summary['pusht']['episodes'] == {'train': 2, 'val': 1, 'test': 1}
    assert summary['cube']['clips']['s1']['train'] == 4


def test_manifest_rejects_leakage_and_label_fields():
    leaked = _manifest()
    leaked['pusht']['split']['test'] = [1]
    with pytest.raises(ValueError, match='leakage'):
        validate_manifest(leaked)
    labelled = _manifest()
    labelled['cube']['s0']['train'][0]['reward'] = 1
    with pytest.raises(ValueError, match='forbidden clip fields'):
        validate_manifest(labelled)


def test_training_columns_are_label_free():
    assert_label_free_columns(('episode_idx', 'pixels', 'action'))
    for forbidden in ('state', 'object_mask', 'reward', 'contact_event', 'success'):
        with pytest.raises(ValueError):
            assert_label_free_columns(('pixels', forbidden))


def test_action_rows_end_before_target():
    context, target, actions = clip_rows({'row': 20}, frameskip=5)
    assert context == 20 and target == 25
    assert actions == [20, 21, 22, 23, 24]
    assert target not in actions


def test_schedule_is_reproducible_and_seed_sensitive():
    first = make_index_schedule(32, 8, 10, 3072)
    second = make_index_schedule(32, 8, 10, 3072)
    other = make_index_schedule(32, 8, 10, 3073)
    assert np.array_equal(first, second)
    assert schedule_sha256(first) == schedule_sha256(second)
    assert not np.array_equal(first, other)


def test_mse_and_l1_have_finite_nonzero_gradients():
    prediction = torch.tensor([1.0, -2.0], requires_grad=True)
    target = torch.zeros(2)
    mse = spatial_prediction_loss(prediction, target, 'mse')
    mse.backward(retain_graph=True)
    mse_grad = prediction.grad.clone()
    prediction.grad.zero_()
    l1 = spatial_prediction_loss(prediction, target, 'l1')
    l1.backward()
    assert torch.isfinite(mse) and torch.isfinite(l1)
    assert float(mse_grad.abs().sum()) > 0
    assert float(prediction.grad.abs().sum()) > 0
    assert not torch.allclose(mse_grad, prediction.grad)


def test_shared_dynamics_forward_backward_for_every_environment():
    torch.manual_seed(0)
    model = _tiny_model()
    widths = (10, 10, 25)
    for env_index, width in enumerate(widths):
        model.zero_grad(set_to_none=True)
        prediction = model(
            torch.randn(2, 1, 4, 8),
            torch.randn(2, 1, width),
            torch.full((2,), env_index, dtype=torch.long),
        )
        prediction.square().mean().backward()
        assert prediction.shape == (2, 1, 4, 8)
        assert model.blocks[0].spatial.in_proj_weight.grad is not None


def test_loss_gradient_calibration_does_not_change_weights():
    torch.manual_seed(1)
    model = _tiny_model()
    before = tensor_state_sha256(model.state_dict())
    batches = []
    for env_index, width in enumerate((10, 10, 25)):
        batches.append(
            (
                torch.randn(2, 1, 4, 8),
                torch.randn(2, 1, width),
                torch.full((2,), env_index, dtype=torch.long),
                torch.randn(2, 1, 4, 8),
            )
        )
    result = calibrate_loss_scales(model, batches)
    assert result['scale']['mse'] == pytest.approx(1.0)
    assert result['scale']['l1'] > 0
    assert tensor_state_sha256(model.state_dict()) == before


def test_checkpoint_save_resume_and_no_overwrite(tmp_path):
    torch.manual_seed(2)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {'arm': 'A', 'steps': 2}
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    path = tmp_path / 'checkpoint_step1.pt'
    save_checkpoint_no_overwrite(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        run_config=config,
    )
    expected = model(torch.ones(2, 3)).detach()
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_resume_checkpoint(path, restored, restored_optimizer, config)
    assert payload['step'] == 1
    assert torch.allclose(restored(torch.ones(2, 3)), expected)
    with pytest.raises(FileExistsError):
        save_checkpoint_no_overwrite(
            path,
            model=model,
            optimizer=optimizer,
            step=1,
            run_config=config,
        )


def test_resume_rejects_changed_conditions(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint_no_overwrite(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        run_config={'seed': 1},
    )
    with pytest.raises(ValueError, match='config'):
        load_resume_checkpoint(path, model, optimizer, {'seed': 2})


def test_common_token_metrics_compare_all_three_baselines():
    current = torch.zeros(2, 4, 3)
    target = current.clone()
    target[:, 0] = 2
    correct = target.clone()
    shuffled = current + 1
    report = token_error_metrics(correct, target, current, shuffled)
    assert set(report) >= {'correct_action', 'persistence', 'action_shuffle'}
    assert report['correct_action']['mse'] == 0
    assert report['persistence']['mse'] > 0
    assert report['correct_action']['moving_quartile_mse'] == 0


class _IdentityProbe:
    kind = 'vector'

    def predict(self, features):
        return np.asarray(features)[:, :2]


def test_probe_gate_blocks_metrics_when_readout_error_exceeds_motion():
    features = np.full((4, 2), 10.0, dtype=np.float32)
    current = np.zeros((4, 2), dtype=np.float32)
    future = np.ones((4, 2), dtype=np.float32)
    report = probe_metric_report(
        _IdentityProbe(),
        true_current_features=features,
        true_future_features=features,
        predicted_features=features,
        shuffled_features=features,
        state_current=current,
        state_future=future,
    )
    assert not report['eligible_for_model_claim']
    assert 'diagnostic_only' in report['interpretation']
    assert report['probe_true_dino_delta_mae'] >= 0
    assert set(report['delta_motion']) == {
        'correct_action_mae',
        'persistence_mae',
        'action_shuffle_mae',
    }


def test_object_motion_challenge_thresholds_are_evaluation_only():
    current = np.zeros((3, 2), dtype=np.float32)
    future = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
    mask = challenge_mask('pusht', 'block_xy', current, future)
    assert mask.tolist() == [False, False, True]


def _load_launcher():
    path = Path(__file__).parents[2] / 'scripts/train/launch_g2_2x2.py'
    spec = importlib.util.spec_from_file_location('g2_launcher_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_builds_four_sequential_commands_and_refuses_existing_root(tmp_path, monkeypatch):
    launcher = _load_launcher()
    manifest = tmp_path / 'manifest.json'
    checkpoint = tmp_path / 'checkpoint.pt'
    manifest.write_text('{}')
    checkpoint.write_bytes(b'x')
    monkeypatch.setattr(launcher, 'MANIFEST', manifest)
    monkeypatch.setattr(launcher, 'CHECKPOINT', checkpoint)
    output = tmp_path / 'runs'
    commands = launcher.build_commands('/venv/stable-wm/bin/python', output, True)
    assert [command[command.index('--arm') + 1] for command in commands] == list('ABCD')
    calibration_files = {
        command[command.index('--loss-calibration-file') + 1] for command in commands
    }
    assert calibration_files == {str(output / 'gradient_calibration.json')}
    launcher.preflight(output, commands)
    output.mkdir()
    with pytest.raises(FileExistsError):
        launcher.preflight(output, commands)


def test_launcher_stops_on_first_failed_arm(tmp_path, monkeypatch):
    launcher = _load_launcher()
    output = tmp_path / 'runs'
    commands = [['python', 'fake.py', '--output-dir', str(output / arm)] for arm in 'ABCD']
    calls = []

    def fail_first(command, log_path):
        calls.append(command)
        return 7

    monkeypatch.setattr(launcher, 'run_command_tee', fail_first)
    assert launcher.run_launcher(output, commands) == 7
    assert len(calls) == 1
    assert (output / 'STATUS').read_text() == 'FAILED arm=A returncode=7\n'


def test_launcher_tees_progress_to_pane_and_log(tmp_path, monkeypatch):
    launcher = _load_launcher()
    pane = io.BytesIO()
    monkeypatch.setattr(launcher, 'sys', SimpleNamespace(stdout=SimpleNamespace(buffer=pane)))
    log = tmp_path / 'run.log'
    command = [sys.executable, '-c', "print('step=1 eta=0s')"]
    assert launcher.run_command_tee(command, log) == 0
    assert pane.getvalue() == b'step=1 eta=0s\n'
    assert log.read_bytes() == pane.getvalue()
