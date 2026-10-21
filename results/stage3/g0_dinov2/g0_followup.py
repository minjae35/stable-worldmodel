"""G0 follow-up: cube yaw ambiguity and time-controlled goal ranking.

Reuses the frozen DINOv2 ViT-S/14 and the G0 episode sampler.
Does not train a world model.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/workspace/stable-worldmodel')
OUT = ROOT / 'results/stage3/g0_dinov2'
spec = importlib.util.spec_from_file_location('g0', OUT / 'run_g0.py')
g0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g0)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def angle_mae_deg(true, pred):
    return float(np.degrees(np.abs(wrap(true - pred))).mean())


def sym_mae_deg(true, pred, period):
    k = np.round(wrap(true - pred) / period)
    return float(np.degrees(np.abs(wrap(true - pred - k * period))).mean())


def green_mask(im):
    r = im[:, :, 0].astype(np.int16)
    g = im[:, :, 1].astype(np.int16)
    b = im[:, :, 2].astype(np.int16)
    return (g > 180) & (g > r + 40) & (g > b + 40)


def spearman(a, b):
    return g0.spearman(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def fit_linear(x_tr, y_tr, x_te, y_te, yaw_tr, yaw_te, period):
    pred_tr = g0.dual_ridge(x_tr, y_tr, x_tr, 10.0).float().cpu().numpy()
    pred_te = g0.dual_ridge(x_tr, y_tr, x_te, 10.0).float().cpu().numpy()
    # sweep a few alphas and keep the best inner-free choice by train/test both reported
    best = None
    for alpha in g0.ALPHAS:
        tr = g0.dual_ridge(x_tr, y_tr, x_tr, alpha).float().cpu().numpy()
        te = g0.dual_ridge(x_tr, y_tr, x_te, alpha).float().cpu().numpy()
        # select alpha by a held-in reconstruction of the angle targets' val via
        # the caller. Here alpha is chosen to minimize train angle error on a
        # slice passed as the first 20% of train rows, which the caller orders
        # so that slice is the inner episodes.
        ang_tr = np.arctan2(tr[:, 0], tr[:, 1]) / period
        err = angle_mae_deg(yaw_tr, ang_tr) if period == 1 else sym_mae_deg(yaw_tr, ang_tr, np.pi / 2)
        if best is None or err < best[0]:
            best = (err, alpha, tr, te)
    _, alpha, tr, te = best
    def decode(pred, yaw):
        ang = np.arctan2(pred[:, 0], pred[:, 1]) / period
        if period == 1:
            return {
                'mae_deg': angle_mae_deg(yaw, ang),
                'sym90_mae_deg': sym_mae_deg(yaw, ang, np.pi / 2),
            }
        return {
            'mae_deg': sym_mae_deg(yaw, ang, np.pi / 2),
            'sym90_mae_deg': sym_mae_deg(yaw, ang, np.pi / 2),
        }
    return {'alpha': alpha, 'train': decode(tr, yaw_tr), 'test': decode(te, yaw_te)}


def train_mlp(x, y, yaw, train_idx, inner_idx, test_idx, hidden=128, epochs=40):
    device = 'cuda'
    xt = x.to(device)
    yt = y.to(device)
    ymu = yt[train_idx].mean(0, keepdim=True)
    ysd = yt[train_idx].std(0, keepdim=True).clamp_min(1e-6)
    xmu = xt[train_idx].mean(0, keepdim=True)
    xsd = xt[train_idx].std(0, keepdim=True).clamp_min(1e-6)
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, 2),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best_state, best_val, wait = None, 1e9, 0
    inner = inner_idx
    for epoch in range(epochs):
        net.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        for s in range(0, len(perm), 256):
            batch = perm[s:s + 256]
            pred = net((xt[batch] - xmu) / xsd)
            target = (yt[batch] - ymu) / ysd
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net((xt[inner] - xmu) / xsd) * ysd + ymu
        ang = torch.atan2(pred[:, 0], pred[:, 1]).cpu().numpy()
        val = angle_mae_deg(yaw[inner.cpu().numpy()], ang)
        if val < best_val - 0.2:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 8:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(idx):
        with torch.no_grad():
            pred = net((xt[idx] - xmu) / xsd) * ysd + ymu
        return torch.atan2(pred[:, 0], pred[:, 1]).cpu().numpy()

    return {
        'epochs_ran': epoch + 1,
        'inner_mae_deg': best_val,
        'train': {
            'mae_deg': angle_mae_deg(yaw[train_idx.cpu().numpy()], predict(train_idx)),
            'sym90_mae_deg': sym_mae_deg(yaw[train_idx.cpu().numpy()], predict(train_idx), np.pi / 2),
        },
        'test': {
            'mae_deg': angle_mae_deg(yaw[test_idx.cpu().numpy()], predict(test_idx)),
            'sym90_mae_deg': sym_mae_deg(yaw[test_idx.cpu().numpy()], predict(test_idx), np.pi / 2),
        },
    }


def train_attention(tokens, y, yaw, train_idx, inner_idx, test_idx, epochs=40):
    device = 'cuda'
    tok = tokens.to(device)
    yt = y.to(device)
    ymu = yt[train_idx].mean(0, keepdim=True)
    ysd = yt[train_idx].std(0, keepdim=True).clamp_min(1e-6)
    d = tok.shape[-1]
    w = torch.nn.Parameter(torch.zeros(d, device=device))
    v = torch.nn.Parameter(torch.zeros(d, 2, device=device))
    torch.nn.init.normal_(v, std=0.02)
    opt = torch.optim.AdamW([w, v], lr=1e-3, weight_decay=1e-3)
    best, best_val, wait = None, 1e9, 0

    def forward(idx):
        h = tok[idx]
        alpha = torch.softmax(h @ w, dim=1)
        pooled = (alpha.unsqueeze(-1) * h).sum(1)
        return pooled @ v

    for epoch in range(epochs):
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        for s in range(0, len(perm), 256):
            batch = perm[s:s + 256]
            pred = forward(batch)
            loss = F.mse_loss(pred, (yt[batch] - ymu) / ysd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = forward(inner_idx) * ysd + ymu
        ang = torch.atan2(pred[:, 0], pred[:, 1]).cpu().numpy()
        val = angle_mae_deg(yaw[inner_idx.cpu().numpy()], ang)
        if val < best_val - 0.2:
            best_val = val
            best = (w.detach().clone(), v.detach().clone())
            wait = 0
        else:
            wait += 1
            if wait >= 8:
                break
    w.data.copy_(best[0])
    v.data.copy_(best[1])

    def predict(idx):
        with torch.no_grad():
            pred = forward(idx) * ysd + ymu
        return torch.atan2(pred[:, 0], pred[:, 1]).cpu().numpy()

    return {
        'epochs_ran': epoch + 1,
        'inner_mae_deg': best_val,
        'train': {
            'mae_deg': angle_mae_deg(yaw[train_idx.cpu().numpy()], predict(train_idx)),
            'sym90_mae_deg': sym_mae_deg(yaw[train_idx.cpu().numpy()], predict(train_idx), np.pi / 2),
        },
        'test': {
            'mae_deg': angle_mae_deg(yaw[test_idx.cpu().numpy()], predict(test_idx)),
            'sym90_mae_deg': sym_mae_deg(yaw[test_idx.cpu().numpy()], predict(test_idx), np.pi / 2),
        },
    }


def local_pool(tokens):
    # tokens (N,P,D). Keep the 8 patches farthest from the per-image median patch.
    med = tokens.median(dim=1).values
    dist = (tokens - med.unsqueeze(1)).pow(2).sum(-1)
    idx = dist.topk(8, dim=1).indices
    gather = idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return tokens.gather(1, gather).mean(1)


def load_rows(dataset, rows, columns):
    table = g0.take_columns(dataset, np.asarray(rows), columns)
    images = np.stack([g0.decode_image(b) for b in table['pixels']])
    return images, table


def pair_distances(cls, tok):
    # cls (P,2,D), tok (P,2,N,D) where 2 is (now, goal)
    cls_d = np.linalg.norm(cls[:, 0] - cls[:, 1], axis=1)
    diff = tok[:, 0] - tok[:, 1]
    patch_mse = (diff ** 2).mean(axis=(1, 2))
    return cls_d, patch_mse


def summarize_rank(name, feature, target):
    return {
        'name': name,
        'spearman': spearman(feature, target),
        'n': int(len(feature)),
    }


def shuffled_spearman(feature_goal_side, other, target, rng):
    # feature_goal_side is the goal token/cls, other is the current.
    vals = []
    for _ in range(30):
        perm = rng.permutation(len(target))
        if other.ndim == 2:
            d = np.linalg.norm(other - feature_goal_side[perm], axis=1)
        else:
            d = ((other - feature_goal_side[perm]) ** 2).mean(axis=(1, 2))
        vals.append(spearman(d, target))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return {'mean': float(vals.mean()), 'std': float(vals.std())}


def main():
    import lance
    from transformers import Dinov2Model

    print('loading dinov2', flush=True)
    dino = Dinov2Model.from_pretrained(g0.WEIGHTS, local_files_only=True).cuda().eval()
    for p in dino.parameters():
        p.requires_grad_(False)

    report = {
        'cube_yaw': {},
        'goal': {},
        'notes': {
            'cube_success_code': 'OGBCube success is ||position - target|| <= 0.04m. Yaw is not in the success test.',
            'cube_appearance': 'Cube geom uses one scalar half-extent (0.02) and a single RGBA. Faces are not textured.',
            'planner_goal': 'scripts/plan/config/{pusht,tworoom,cube}.yaml set eval.goal_offset_steps: 25. The goal observation is a later dataset frame, not a separately rendered target image.',
            'pusht_goal_measurement': 'Green marker centroid, converted by pixel = state * 224/512 (no y flip; checked against the blue agent). Angle of the marker was not used: template overlap did not identify a unique heading.',
        },
    }

    # ---------- cube yaw ----------
    print('cube yaw sample', flush=True)
    dataset = lance.dataset(g0.PATHS['cube'])
    starts, ends = g0.episode_bounds(dataset)
    episodes = g0.choose_episodes(starts, ends)
    rows, ep_ids, steps = g0.sample_rows(starts, ends, episodes)
    images, table = load_rows(
        dataset, rows,
        ['pixels', 'privileged_block_0_yaw', 'privileged_block_0_pos',
         'privileged_target_block_pos', 'success'],
    )
    cls, tok = g0.encode_dino(dino, images, 224)
    yaw = g0.as_float_array(table['privileged_block_0_yaw'])[:, 0]
    pos = g0.as_float_array(table['privileged_block_0_pos'])
    target = g0.as_float_array(table['privileged_target_block_pos'])
    success = g0.as_float_array(table['success'])[:, 0]
    pos_err = np.linalg.norm(pos - target, axis=1)
    report['cube_yaw']['success_matches_4cm'] = float(np.mean((success > 0.5) == (pos_err <= 0.04)))
    report['cube_yaw']['success_rate'] = float(success.mean())

    test_eps = episodes[::4]
    train_eps = np.setdiff1d(episodes, test_eps)
    inner_eps = train_eps[::5]
    fit_eps = np.setdiff1d(train_eps, inner_eps)
    # order train as inner-then-rest so linear alpha selection can use the prefix
    def mask(eps):
        return torch.as_tensor(np.flatnonzero(np.isin(ep_ids, eps)), device='cuda')

    tr = mask(fit_eps)
    inner = mask(inner_eps)
    te = mask(test_eps)
    all_tr = torch.cat([inner, tr])
    pool4 = g0.pool_tokens(tok.float(), 4)
    local = local_pool(tok.float())
    targets = {
        'sincos': (np.stack([np.sin(yaw), np.cos(yaw)], 1), 1.0),
        'sin2cos2': (np.stack([np.sin(2 * yaw), np.cos(2 * yaw)], 1), 2.0),
        'sin4cos4': (np.stack([np.sin(4 * yaw), np.cos(4 * yaw)], 1), 4.0),
    }
    features = {
        'cls': cls.float(),
        'spatial_pool4': pool4,
        'local_top8': local,
    }
    linear = {}
    for feat_name, feat in features.items():
        linear[feat_name] = {}
        for tgt_name, (arr, period) in targets.items():
            y = torch.from_numpy(arr.astype(np.float32))
            # alpha selected on inner episodes, refit on all train
            x = feat.cuda()
            y_g = y.cuda()
            best_alpha, best_val = None, 1e9
            # sin(kθ) identifies θ modulo 2π/k. k=1,2,4 -> periods 2π, π, π/2.
            fold = None if period == 1 else (2 * np.pi / period)
            for alpha in g0.ALPHAS:
                pred = g0.dual_ridge(x[tr], y_g[tr], x[inner], alpha)
                ang = torch.atan2(pred[:, 0], pred[:, 1]).cpu().numpy() / period
                yw_inner = yaw[inner.cpu().numpy()]
                err = angle_mae_deg(yw_inner, ang) if fold is None else sym_mae_deg(yw_inner, ang, fold)
                if err < best_val:
                    best_val, best_alpha = err, alpha
            pred_tr = g0.dual_ridge(x[all_tr], y_g[all_tr], x[all_tr], best_alpha).cpu().numpy()
            pred_te = g0.dual_ridge(x[all_tr], y_g[all_tr], x[te], best_alpha).cpu().numpy()
            def pack(pred, idx):
                ang = np.arctan2(pred[:, 0], pred[:, 1]) / period
                yw = yaw[idx.cpu().numpy()]
                return {
                    'mae_deg': angle_mae_deg(yw, ang) if fold is None else sym_mae_deg(yw, ang, fold),
                    'sym90_mae_deg': sym_mae_deg(yw, ang, np.pi / 2),
                }
            # For period!=1, atan2/period is the folded angle. sym90 compares it to raw yaw.
            linear[feat_name][tgt_name] = {
                'alpha': best_alpha,
                'inner_mae_deg': best_val,
                'train_r2': float(np.nanmean(g0.r2_score(arr[all_tr.cpu().numpy()], pred_tr))),
                'test_r2': float(np.nanmean(g0.r2_score(arr[te.cpu().numpy()], pred_te))),
                'train': pack(pred_tr, all_tr),
                'test': pack(pred_te, te),
            }
            print(feat_name, tgt_name, 'test', linear[feat_name][tgt_name]['test'], 'r2', round(linear[feat_name][tgt_name]['test_r2'], 3), flush=True)
    y_sc = torch.from_numpy(targets['sincos'][0].astype(np.float32))
    mlp = train_mlp(pool4, y_sc, yaw, all_tr, inner, te)
    attn = train_attention(tok.float(), y_sc, yaw, all_tr, inner, te)
    # constant baseline: train circular mean
    mean_ang = np.angle(np.exp(1j * yaw[all_tr.cpu().numpy()]).mean())
    base = np.full(len(te), mean_ang)
    report['cube_yaw']['linear'] = linear
    report['cube_yaw']['mlp_pool4'] = mlp
    report['cube_yaw']['attention_readout'] = attn
    report['cube_yaw']['constant_test_mae_deg'] = angle_mae_deg(yaw[te.cpu().numpy()], base)
    report['cube_yaw']['constant_test_sym90_mae_deg'] = sym_mae_deg(yaw[te.cpu().numpy()], base, np.pi / 2)
    report['cube_yaw']['chance_note'] = 'Uniform yaw has absolute MAE 90 deg and 90-degree-symmetry MAE 22.5 deg.'
    print('mlp', mlp, flush=True)
    print('attn', attn, flush=True)
    del images, tok, cls
    torch.cuda.empty_cache()

    # ---------- fixed-offset goal pairs ----------
    goal_xy = np.array([226.93271166, 283.85171816])  # median of unoccluded green centroids, 512 space
    rng = np.random.default_rng(0)
    for name in ('pusht', 'tworoom', 'cube'):
        print('pairs', name, flush=True)
        ds = lance.dataset(g0.PATHS[name])
        st, en = g0.episode_bounds(ds)
        eps = g0.choose_episodes(st, en)[::4]
        pairs = []
        for ep in eps:
            length = int(en[ep] - st[ep])
            valid = length - 25
            if valid < 8:
                continue
            for frac in (0.15, 0.35, 0.55, 0.75):
                s = int(frac * (valid - 1))
                pairs.append((int(st[ep] + s), int(st[ep] + s + 25), int(ep)))
        now_rows = [p[0] for p in pairs]
        goal_rows = [p[1] for p in pairs]
        cols = {
            'pusht': ['pixels', 'state'],
            'tworoom': ['pixels', 'pos_agent', 'distance_to_target'],
            'cube': ['pixels', 'privileged_block_0_pos', 'privileged_block_0_yaw',
                     'privileged_target_block_pos', 'success'],
        }[name]
        img_now, tab_now = load_rows(ds, now_rows, cols)
        img_goal, tab_goal = load_rows(ds, goal_rows, cols)
        both = np.concatenate([img_now, img_goal], 0)
        cls_b, tok_b = g0.encode_dino(dino, both, 224)
        n = len(pairs)
        cls_p = cls_b.numpy().reshape(2, n, -1).transpose(1, 0, 2)
        side = int(round(tok_b.shape[1] ** 0.5))
        tok_p = tok_b.numpy().reshape(2, n, side * side, -1).transpose(1, 0, 2, 3)
        cls_d, patch_mse = pair_distances(cls_p, tok_p)
        out = {'n_pairs': n, 'delta_steps': 25}
        if name == 'pusht':
            s0 = g0.as_float_array(tab_now['state'])
            s1 = g0.as_float_array(tab_goal['state'])
            block_d = np.linalg.norm(s0[:, 2:4] - s1[:, 2:4], axis=1)
            ang_d = np.abs(wrap(s0[:, 4] - s1[:, 4]))
            agent_d = np.linalg.norm(s0[:, :2] - s1[:, :2], axis=1)
            pos_err0 = np.linalg.norm(s0[:, 2:4] - goal_xy, axis=1)
            pos_err1 = np.linalg.norm(s1[:, 2:4] - goal_xy, axis=1)
            out['to_future_frame'] = {
                'patch_mse_vs_block_pos': summarize_rank('patch', patch_mse, block_d),
                'cls_vs_block_pos': summarize_rank('cls', cls_d, block_d),
                'patch_mse_vs_block_angle': summarize_rank('patch', patch_mse, ang_d),
                'patch_mse_vs_agent_pos': summarize_rank('patch', patch_mse, agent_d),
                'shuffle_patch_vs_block_pos': shuffled_spearman(tok_p[:, 1], tok_p[:, 0], block_d, rng),
            }
            out['to_green_goal'] = {
                'goal_xy_512': goal_xy.tolist(),
                'median_pos_err': float(np.median(pos_err0)),
                'frac_block_within_20': float(np.mean(pos_err0 < 20)),
                'patch_mse_vs_drop_in_pos_err': summarize_rank('patch', patch_mse, pos_err0 - pos_err1),
                'patch_mse_vs_current_pos_err': summarize_rank('patch', patch_mse, pos_err0),
            }
        elif name == 'tworoom':
            a0 = g0.as_float_array(tab_now['pos_agent'])
            a1 = g0.as_float_array(tab_goal['pos_agent'])
            d0 = g0.as_float_array(tab_now['distance_to_target'])[:, 0]
            d1 = g0.as_float_array(tab_goal['distance_to_target'])[:, 0]
            move = np.linalg.norm(a0 - a1, axis=1)
            out['to_future_frame'] = {
                'patch_mse_vs_agent_move': summarize_rank('patch', patch_mse, move),
                'cls_vs_agent_move': summarize_rank('cls', cls_d, move),
                'patch_mse_vs_target_progress': summarize_rank('patch', patch_mse, d0 - d1),
                'patch_mse_vs_abs_target_change': summarize_rank('patch', patch_mse, np.abs(d0 - d1)),
                'shuffle_patch_vs_agent_move': shuffled_spearman(tok_p[:, 1], tok_p[:, 0], move, rng),
                'agent_move_vs_target_progress': summarize_rank('move', move, d0 - d1),
            }
        else:
            p0 = g0.as_float_array(tab_now['privileged_block_0_pos'])
            p1 = g0.as_float_array(tab_goal['privileged_block_0_pos'])
            y0 = g0.as_float_array(tab_now['privileged_block_0_yaw'])[:, 0]
            y1 = g0.as_float_array(tab_goal['privileged_block_0_yaw'])[:, 0]
            tpos = g0.as_float_array(tab_now['privileged_target_block_pos'])
            suc0 = g0.as_float_array(tab_now['success'])[:, 0]
            suc1 = g0.as_float_array(tab_goal['success'])[:, 0]
            move = np.linalg.norm(p0 - p1, axis=1)
            yaw_d = np.abs(wrap(y0 - y1))
            yaw90 = np.array([min(np.abs(wrap(a)) , np.pi / 2 - np.abs(wrap(a)) if False else min(abs(wrap(a)) % (np.pi / 2), np.pi / 2 - (abs(wrap(a)) % (np.pi / 2)))) for a in (y0 - y1)])
            err0 = np.linalg.norm(p0 - tpos, axis=1)
            out['to_future_frame'] = {
                'patch_mse_vs_block_move': summarize_rank('patch', patch_mse, move),
                'cls_vs_block_move': summarize_rank('cls', cls_d, move),
                'patch_mse_vs_yaw': summarize_rank('patch', patch_mse, yaw_d),
                'patch_mse_vs_yaw90': summarize_rank('patch', patch_mse, yaw90),
                'patch_mse_vs_pos_err_drop': summarize_rank('patch', patch_mse, err0 - np.linalg.norm(p1 - tpos, axis=1)),
                'patch_mse_vs_success_gain': summarize_rank('patch', patch_mse, suc1 - suc0),
                'shuffle_patch_vs_block_move': shuffled_spearman(tok_p[:, 1], tok_p[:, 0], move, rng),
            }
            out['success_rate_now'] = float(suc0.mean())
        report['goal'][name] = out
        print(json.dumps(out)[:800], flush=True)
        del img_now, img_goal, both, cls_b, tok_b

    with open(OUT / 'followup.json', 'w') as f:
        json.dump(report, f)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
