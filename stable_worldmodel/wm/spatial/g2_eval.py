"""Evaluation-only diagnostics for G2 spatial dynamics.

State labels are deliberately isolated here. They may select/report evaluation
subsets and test frozen-DINO readout quality, but never affect G2 sampling,
losses, gradients, or optimizer updates.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from stable_worldmodel.wm.spatial.g2 import _take, clip_rows

EVAL_LABEL_COLUMNS = {
    'pusht': ('state',),
    'tworoom': ('pos_agent',),
    'cube': ('privileged_block_0_pos', 'proprio_effector_pos'),
}


def token_error_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    shuffled_prediction: torch.Tensor,
) -> dict:
    """Common MSE/L1 metrics plus motion-focused patch diagnostics."""
    if not (
        prediction.shape == target.shape == current.shape == shuffled_prediction.shape
    ):
        raise ValueError('all token tensors must have the same shape')
    target = target.detach()
    current = current.detach()
    per_patch_motion = (target - current).float().square().mean(dim=-1)
    threshold = torch.quantile(per_patch_motion, 0.75, dim=-1, keepdim=True)
    moving = per_patch_motion >= threshold
    weights = per_patch_motion / per_patch_motion.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def errors(value):
        delta = value.float() - target.float()
        mse_patch = delta.square().mean(dim=-1)
        l1_patch = delta.abs().mean(dim=-1)
        return {
            'mse': float(mse_patch.mean()),
            'l1': float(l1_patch.mean()),
            'moving_quartile_mse': float(mse_patch[moving].mean()),
            'moving_quartile_l1': float(l1_patch[moving].mean()),
            'motion_weighted_mse': float((mse_patch * weights).sum(dim=-1).mean()),
            'motion_weighted_l1': float((l1_patch * weights).sum(dim=-1).mean()),
        }

    return {
        'correct_action': errors(prediction),
        'persistence': errors(current),
        'action_shuffle': errors(shuffled_prediction),
        'motion_energy_mean': float(per_patch_motion.mean()),
    }


def pool4(tokens: torch.Tensor) -> np.ndarray:
    """Pool 16x16 DINO tokens to a 4x4 spatial readout feature."""
    if tokens.ndim != 3 or tokens.shape[1] != 256:
        raise ValueError(f'expected (N,256,D), got {tuple(tokens.shape)}')
    grid = tokens.float().reshape(tokens.shape[0], 16, 16, tokens.shape[-1])
    grid = grid.permute(0, 3, 1, 2)
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (4, 4))
    return pooled.permute(0, 2, 3, 1).reshape(tokens.shape[0], -1).cpu().numpy()


def load_state_labels(dataset, clips: list[dict], env: str, frameskip: int) -> dict:
    """Load state labels for evaluation after training data is finalized."""
    columns = EVAL_LABEL_COLUMNS[env]
    rows = []
    pairs = []
    for clip in clips:
        context, target, _ = clip_rows(clip, frameskip)
        rows.extend([context, target])
        pairs.append((context, target))
    _, by_row = _take(dataset, rows, columns)

    def vector(value):
        return np.asarray(value, dtype=np.float32).reshape(-1)

    if env == 'pusht':
        now = np.stack([vector(by_row[a]['state']) for a, _ in pairs])
        future = np.stack([vector(by_row[b]['state']) for _, b in pairs])
        return {
            'agent_xy': (now[:, :2], future[:, :2]),
            'block_xy': (now[:, 2:4], future[:, 2:4]),
            'block_angle': (now[:, 4], future[:, 4]),
        }
    if env == 'tworoom':
        now = np.stack([vector(by_row[a]['pos_agent']) for a, _ in pairs])
        future = np.stack([vector(by_row[b]['pos_agent']) for _, b in pairs])
        return {'agent_xy': (now, future)}
    block_now = np.stack(
        [vector(by_row[a]['privileged_block_0_pos']) for a, _ in pairs]
    )
    block_future = np.stack(
        [vector(by_row[b]['privileged_block_0_pos']) for _, b in pairs]
    )
    effector_now = np.stack(
        [vector(by_row[a]['proprio_effector_pos']) for a, _ in pairs]
    )
    effector_future = np.stack(
        [vector(by_row[b]['proprio_effector_pos']) for _, b in pairs]
    )
    return {
        'block_xyz': (block_now, block_future),
        'effector_xyz': (effector_now, effector_future),
    }


def _angle_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.stack([np.sin(values), np.cos(values)], axis=1)


def _angle_from_features(values: np.ndarray) -> np.ndarray:
    return np.arctan2(values[:, 0], values[:, 1])


def state_distance(prediction, target, kind: str) -> np.ndarray:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if kind == 'angle':
        delta = np.arctan2(np.sin(prediction - target), np.cos(prediction - target))
        return np.degrees(np.abs(delta).reshape(-1))
    return np.linalg.norm(prediction - target, axis=-1)


def state_motion(current, future, kind: str) -> np.ndarray:
    return state_distance(future, current, kind)


def state_delta(current, future, kind: str) -> np.ndarray:
    delta = np.asarray(future) - np.asarray(current)
    if kind == 'angle':
        return np.arctan2(np.sin(delta), np.cos(delta))
    return delta


def challenge_mask(env: str, target: str, current, future) -> np.ndarray:
    """Evaluation-only object-motion challenge selector."""
    motion = state_motion(current, future, 'angle' if target == 'block_angle' else 'vector')
    thresholds = {
        ('pusht', 'agent_xy'): 15.0,
        ('pusht', 'block_xy'): 15.0,
        ('pusht', 'block_angle'): 10.0,
        ('tworoom', 'agent_xy'): 15.0,
        ('cube', 'block_xyz'): 0.02,
        ('cube', 'effector_xyz'): 0.02,
    }
    return motion >= thresholds[(env, target)]


class RidgeProbe:
    def __init__(self, model, x_mean, x_std, kind: str, alpha: float):
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std
        self.kind = kind
        self.alpha = float(alpha)

    def predict(self, features: np.ndarray) -> np.ndarray:
        value = self.model.predict((features - self.x_mean) / self.x_std)
        return _angle_from_features(value) if self.kind == 'angle' else value


class ArrayLinearModel:
    """Minimal serializable linear predictor used by the CUDA ridge solver."""

    def __init__(self, weight: np.ndarray, intercept: np.ndarray):
        self.weight = np.asarray(weight)
        self.intercept = np.asarray(intercept)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features) @ self.weight + self.intercept


def fit_probe(
    train_features: np.ndarray,
    train_target: np.ndarray,
    val_features: np.ndarray,
    val_target: np.ndarray,
    kind: str,
    solver: str = 'lsqr',
) -> tuple[RidgeProbe, dict]:
    """Fit ridge on true DINO features and select alpha only on validation."""
    from sklearn.linear_model import Ridge

    x_mean = train_features.mean(axis=0, keepdims=True)
    x_std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-6)
    x_train = (train_features - x_mean) / x_std
    x_val = (val_features - x_mean) / x_std
    y_train = _angle_features(train_target) if kind == 'angle' else train_target
    best = None
    rows = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha, solver=solver, tol=1e-5, max_iter=2000)
        model.fit(x_train, y_train)
        raw = model.predict(x_val)
        prediction = _angle_from_features(raw) if kind == 'angle' else raw
        mae = float(state_distance(prediction, val_target, kind).mean())
        rows.append({'alpha': alpha, 'val_mae': mae})
        if best is None or mae < best[0]:
            best = (mae, alpha, model)
    probe = RidgeProbe(best[2], x_mean, x_std, kind, best[1])
    return probe, {
        'solver': f'sklearn_{solver}',
        'alpha_search': rows,
        'chosen_alpha': best[1],
        'val_mae': best[0],
    }


def fit_probe_cuda_dual(
    train_features: np.ndarray,
    train_target: np.ndarray,
    val_features: np.ndarray,
    val_target: np.ndarray,
    kind: str,
    device: str = 'cuda:0',
) -> tuple[RidgeProbe, dict]:
    """Exact float64 dual ridge with alpha selected only on validation."""
    x_mean = train_features.mean(axis=0, keepdims=True)
    x_std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-6)
    x_train_np = (train_features - x_mean) / x_std
    x_val_np = (val_features - x_mean) / x_std
    y_train_np = _angle_features(train_target) if kind == 'angle' else train_target
    if y_train_np.ndim == 1:
        y_train_np = y_train_np[:, None]

    x_train = torch.as_tensor(x_train_np, dtype=torch.float64, device=device)
    x_val = torch.as_tensor(x_val_np, dtype=torch.float64, device=device)
    y_train = torch.as_tensor(y_train_np, dtype=torch.float64, device=device)
    x_offset = x_train.mean(dim=0, keepdim=True)
    y_offset = y_train.mean(dim=0, keepdim=True)
    x_centered = x_train - x_offset
    y_centered = y_train - y_offset
    kernel = x_centered @ x_centered.T
    identity = torch.eye(kernel.shape[0], dtype=kernel.dtype, device=device)
    best = None
    rows = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        dual = torch.linalg.solve(kernel + alpha * identity, y_centered)
        weight = x_centered.T @ dual
        intercept = y_offset - x_offset @ weight
        raw = (x_val @ weight + intercept).float().cpu().numpy()
        prediction = _angle_from_features(raw) if kind == 'angle' else raw
        mae = float(state_distance(prediction, val_target, kind).mean())
        rows.append({'alpha': alpha, 'val_mae': mae})
        if best is None or mae < best[0]:
            best = (
                mae,
                alpha,
                weight.float().cpu().numpy(),
                intercept.float().cpu().numpy().reshape(-1),
            )
    model = ArrayLinearModel(best[2], best[3])
    probe = RidgeProbe(model, x_mean, x_std, kind, best[1])
    return probe, {
        'solver': 'torch_cuda_dual_float64',
        'alpha_search': rows,
        'chosen_alpha': best[1],
        'val_mae': best[0],
    }


def probe_metric_report(
    probe: RidgeProbe,
    *,
    true_current_features: np.ndarray,
    true_future_features: np.ndarray,
    predicted_features: np.ndarray,
    shuffled_features: np.ndarray,
    state_current: np.ndarray,
    state_future: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Gate state conclusions on true-DINO probe accuracy versus motion."""
    if mask is None:
        mask = np.ones(len(state_current), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return {'n': 0, 'eligible_for_model_claim': False, 'reason': 'empty subset'}
    kind = probe.kind
    decoded_true_future = probe.predict(true_future_features)
    decoded_current = probe.predict(true_current_features)
    decoded_prediction = probe.predict(predicted_features)
    decoded_shuffle = probe.predict(shuffled_features)
    probe_absolute_error = float(
        state_distance(decoded_true_future[mask], state_future[mask], kind).mean()
    )
    true_delta = state_delta(state_current, state_future, kind)
    decoded_true_delta = state_delta(decoded_current, decoded_true_future, kind)
    decoded_prediction_delta = state_delta(decoded_current, decoded_prediction, kind)
    decoded_shuffle_delta = state_delta(decoded_current, decoded_shuffle, kind)
    probe_delta_error = float(
        state_distance(decoded_true_delta[mask], true_delta[mask], kind).mean()
    )
    motion = state_motion(state_current[mask], state_future[mask], kind)
    mean_motion = float(motion.mean())
    eligible = bool(
        probe_absolute_error < mean_motion and probe_delta_error < mean_motion
    )
    report = {
        'n': int(mask.sum()),
        'probe_true_dino_mae': probe_absolute_error,
        'probe_true_dino_delta_mae': probe_delta_error,
        'mean_true_motion': mean_motion,
        'eligible_for_model_claim': eligible,
        'gate': (
            'max(probe_true_dino_mae, probe_true_dino_delta_mae) '
            '< mean_true_motion'
        ),
        'correct_action_mae': float(
            state_distance(decoded_prediction[mask], state_future[mask], kind).mean()
        ),
        'persistence_mae': float(
            state_distance(decoded_current[mask], state_future[mask], kind).mean()
        ),
        'action_shuffle_mae': float(
            state_distance(decoded_shuffle[mask], state_future[mask], kind).mean()
        ),
        'delta_motion': {
            'correct_action_mae': float(
                state_distance(
                    decoded_prediction_delta[mask], true_delta[mask], kind
                ).mean()
            ),
            'persistence_mae': mean_motion,
            'action_shuffle_mae': float(
                state_distance(
                    decoded_shuffle_delta[mask], true_delta[mask], kind
                ).mean()
            ),
        },
    }
    if not eligible:
        report['interpretation'] = (
            'diagnostic_only: probe error is not below the motion being measured'
        )
    return report


def finite_report(value) -> bool:
    if isinstance(value, dict):
        return all(finite_report(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_report(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True
