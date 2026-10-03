"""Frozen values must match scripts/rdwm/experiment-1.md."""

from rdwm.exp1.config import check_envs, env_order, env_tag, run_dir


def test_frozen_architecture(cfg):
    m = cfg['model']
    assert (m['image_size'], m['patch_size']) == (224, 14)
    assert (m['encoder_width'], m['encoder_depth'], m['encoder_heads'], m['encoder_mlp']) == (192, 12, 3, 768)
    assert ((m['image_size'] // m['patch_size']) ** 2) == 256
    assert ((m['image_size'] // m['patch_size'] // m['pool']) ** 2) == 64
    assert m['projector_hidden'] == 768 and m['head_hidden'] == 768
    assert m['action_block'] == 5 and m['action_hidden'] == 192 and m['dim'] == 192
    assert (m['dyn_depth'], m['dyn_heads'], m['dyn_head_dim'], m['dyn_ffn']) == (6, 3, 64, 768)
    assert m['dyn_dropout'] == 0.1 and m['max_frames'] == 3
    assert m['adapter_rank'] == 32 and m['adapter_blocks'] == [5, 6]


def test_frozen_data_train_eval(cfg):
    d, t, c, e = cfg['data'], cfg['train'], cfg['cem'], cfg['eval']
    assert d['num_obs'] == 5 and d['obs_stride'] == 5
    assert d['history_probs'] == {3: 0.8, 2: 0.1, 1: 0.1}
    assert d['action_std_floor'] == 1e-3
    assert t['batch_size'] == 32 and t['lr'] == 5e-5 and t['weight_decay'] == 1e-3
    assert t['warmup_frac'] == 0.01 and t['grad_clip'] == 1.0 and t['bf16'] is True
    assert t['loss_weights'] == {'t1': 0.5, 't2': 0.5, 'sigreg': 0.09}
    assert t['sigreg'] == {'num_proj': 1024, 'knots': 17, 'positions': 8}
    assert t['budgets'] == {'A': 10000, 'B': 70000, 'C': 70000, 'D': 70000}
    assert len(t['seeds']) == 3
    assert (c['num_samples'], c['n_steps'], c['topk'], c['horizon'], c['receding_horizon'], c['history']) == (300, 30, 30, 5, 1, 3)
    assert (e['goal_offset'], e['exec_budget'], e['test_pairs'], e['pilot_val_pairs']) == (25, 50, 100, 20)


def test_variants_and_envs(cfg):
    assert env_order(cfg) == ['pusht', 'tworoom', 'cube', 'scene', 'reacher', 'pointmaze', 'antmaze']
    v = cfg['variants']
    assert v['A']['single_env'] and not v['A']['adapters'] and not v['A']['pooled']
    assert not v['B']['adapters'] and not v['B']['pooled']
    assert v['C']['adapters'] and not v['C']['pooled']
    assert v['D']['pooled'] and not v['D']['adapters']
    dims = {k: e['action_dim'] for k, e in cfg['envs'].items()}
    assert dims == {'pusht': 2, 'tworoom': 2, 'cube': 5, 'scene': 5, 'reacher': 2, 'pointmaze': 2, 'antmaze': 8}


def test_run_dirs_do_not_collide(cfg):
    all7 = env_order(cfg)
    dirs = {run_dir(cfg, 'full', v, s, all7) for v in 'BCD' for s in (0, 1, 2)}
    dirs |= {run_dir(cfg, 'full', 'A', s, [e]) for s in (0, 1, 2) for e in all7}
    assert len(dirs) == 3 * 3 + 3 * 7
    assert env_tag(cfg, ['cube', 'pusht']) == 'pusht+cube'
    assert check_envs(cfg, 'B', ['cube', 'pusht']) == ['pusht', 'cube']
