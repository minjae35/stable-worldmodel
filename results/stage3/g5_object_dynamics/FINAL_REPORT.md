# Stable-WM Stage 3 G5 final report

## Outcome

G5 implemented and evaluated two scalable, label-free objective changes on the
unchanged H1 Shared SpatialDynamics model: locally motion-balanced spatial loss
and detached-negative action ranking. Four controlled arms continued from the
same H1 step-2,000 model and optimizer for exactly 500 updates each (2,000 new
optimizer updates total). All checkpoints, logs, common evaluations, corrected
post-norm object evaluations, and episode-paired bootstrap artifacts were
preserved.

The result is **not sufficient to claim that G5 solved object-centric dynamics**.
The combined arm D has the best macro feature MSE, but its advantage over the
uniform continuation is small and statistically supported only on TwoRoom.
PushT T-block rotation remains worse than persistence in every arm, and correct
actions do not consistently beat shuffled actions for T-block translation.

## 1. G0-G4 evidence and G5 hypotheses

The authoritative G4 diagnostic is the corrected post-final-LayerNorm analysis;
the earlier mixed pre-/post-norm artifacts remain preserved as `SUPERSEDED`.
Frozen DINOv2-S/14 contains localized T-block information: the applicable L12
post-norm object crop has much lower readout error than the old global probe.
Nevertheless, H1/H3/G4 dynamics all failed to beat persistence on reliable
PushT rotation. H1 remained the primary generalist baseline because H3 and the
G4 fixed temporal encoding did not improve the three-environment macro result.

G5 therefore tested three hypotheses without changing architecture:

1. **Spatial loss imbalance:** uniform patch MSE underweights small localized
   motion. Arm B uses bounded, locally normalized motion weights derived only
   from frozen feature change. Every 4x4 spatial tile retains mean weight 1;
   the uniform component prevents static regions from being discarded.
2. **Action insensitivity:** correct actions need to outperform an in-batch
   action derangement. Arm C adds a factual ranking hinge. The shuffled branch
   is detached, so the objective cannot be satisfied by deliberately worsening
   the negative prediction and does not assert a false counterfactual target.
3. **Interaction objective:** arm D combines the spatial and action terms. It is
   an objective-level proxy for localized action-object interaction, not a claim
   that explicit objects or contacts were discovered.

Training used only RGB-derived frozen features, actions, and episode indices.
State, object masks, rewards, contacts, events, and environment-specific object
definitions were absent from sampling, loss, gradients, and optimizer updates.
They were used only in the locked Development evaluation.

## 2. Controlled setup and validation

| condition | value |
|---|---|
| source | H1 `checkpoint_step2000.pt` |
| architecture | unchanged Shared SpatialDynamics, history 1, no temporal encoding |
| encoder | frozen DINOv2-S/14, final post-norm 16x16 patch tokens |
| seed | 3072 |
| optimizer | restored AdamW, LR 1e-3, weight decay 1e-4; no scheduler |
| batch | 16/environment, balanced PushT/TwoRoom/Cube |
| continuation budget | 500 updates/arm; cumulative step 2,500 |
| action alignment | frameskip 5, action `[t,t+5)`, target `t+5` |
| evaluation | G3 Development cohort; existing final holdout not reused |

All arms have the same source model hash, complete step-2,000 optimizer state,
restored source RNG, train feature cache, anchor schedule, deterministic
no-fixed-point action derangements, split, and update budget. The first ten
updates of each final run formed the smoke/resume test and are included in the
500-update budget.

Initial gradient norm was 0.206772 for uniform MSE. The calibrated objective
scales were A/B/C/D = 1.0000/0.8927/0.8625/0.7874. The C/D ranking lambdas were
0.8901/0.8512, selected to give the ranking component a conservative 25% of the
reference initial-gradient budget before final total-gradient normalization.

The first preflight attempt stopped before any optimizer update because a NumPy
schedule could not be loaded by PyTorch 2.6 `weights_only=True`. Its logs were
preserved under `preflight_failure_numpy_schedule`. The schedule was converted
to a hash-validated tensor-only artifact; arbitrary pickle loading was not
enabled.

## 3. Objectives

| arm | base spatial loss | action term |
|---|---|---|
| A | uniform patch MSE | none |
| B | locally motion-balanced patch MSE | none |
| C | uniform patch MSE | detached-negative factual ranking |
| D | locally motion-balanced patch MSE | detached-negative factual ranking |

Motion weights use only `||DINO(x[t+5])-DINO(x[t])||`, are normalized within
4x4 token tiles, retain a 50% uniform component, and are bounded. No learned
selector, mask, state label, contact label, or environment-specific threshold is
used. This is scalable to additional environments with RGB and actions, but it
does not yet provide explicit persistent object grouping.

## 4. Training stability and compute

| arm | updates | measured step time | launcher wall time (smoke + resume) | peak VRAM | status |
|---|---:|---:|---:|---:|---|
| A uniform | 500 | 65.00 s | 84.30 s | 2.35 GiB | success |
| B motion | 500 | 67.00 s | 85.96 s | 2.35 GiB | success |
| C action | 500 | 71.88 s | 90.26 s | 3.03 GiB | success |
| D combined | 500 | 70.86 s | 89.26 s | 3.03 GiB | success |

No NaN, Inf, OOM, or unsafe interruption occurred. Every final checkpoint is at
step 2,500 and every AdamW parameter state is also at step 2,500. Checkpoints
were saved at cumulative steps 2,010, 2,100, 2,200, 2,300, 2,400, and 2,500.
Common plus corrected-object evaluation of H1 and four G5 models took 104.90 s.

## 5. Common feature evaluation

Lower is better. Macro is the unweighted mean over the three environments.

| model | PushT MSE / MAE | TwoRoom MSE / MAE | Cube MSE / MAE | macro MSE |
|---|---:|---:|---:|---:|
| H1 source | 0.217953 / 0.324016 | 0.122204 / 0.244764 | 0.516093 / 0.500232 | 0.285417 |
| A uniform | **0.214703** / 0.320343 | 0.112457 / 0.235354 | 0.508323 / 0.493903 | 0.278494 |
| B motion | 0.218858 / 0.323721 | 0.121412 / 0.242305 | 0.507300 / 0.494450 | 0.282523 |
| C action | 0.212933 / **0.318483** | 0.110212 / 0.233657 | 0.511670 / 0.495497 | 0.278272 |
| D combined | 0.214434 / 0.320178 | **0.109427** / **0.233129** | **0.506224** / **0.493551** | **0.276695** |

Relative to A, D improves macro MSE by 0.001799 (0.65%). Episode-paired
bootstrap differences (challenger minus A; negative favors challenger) are:

| comparison | PushT MSE delta [95% CI] | TwoRoom | Cube |
|---|---:|---:|---:|
| B - A | +0.003800 [+0.000655,+0.006842] | +0.009238 [+0.004301,+0.015775] | -0.000789 [-0.004308,+0.002461] |
| C - A | -0.002206 [-0.004720,+0.000267] | -0.002143 [-0.005001,+0.000590] | +0.003508 [-0.001220,+0.008130] |
| D - A | -0.000348 [-0.002425,+0.001709] | **-0.003397 [-0.006196,-0.000830]** | -0.001889 [-0.005708,+0.001647] |

The bootstrap unit is the episode (64 episodes/environment, 10,000 paired
resamples). This distinguishes within-cohort episode uncertainty from seed
uncertainty; there is still only one training seed, so no across-seed
reproducibility claim is made.

Motion balancing alone improves Cube's moving-quartile MSE (1.203877 to
1.187870) but worsens PushT and TwoRoom moving-quartile MSE and significantly
worsens their overall feature error. Its static-quartile error also increases
significantly on PushT and TwoRoom. The spatial-imbalance hypothesis is thus not
supported as a general default by arm B.

## 6. Action, state, and corrected object diagnostics

All feature models beat feature-space persistence. Action-shuffle MSE minus
correct-action MSE for A is 0.1181/0.3163/0.4565 on PushT/TwoRoom/Cube. Arm C
changes these gaps to 0.1139/0.3224/0.4463: it increases separation only on
TwoRoom and decreases it on PushT and Cube. D is also inconsistent. The
action-ranking loss therefore does not establish general action grounding.

Challenge metrics below pass their applicable probe gate; natural small-motion
state metrics are `diagnostic_only` where probe error exceeds motion.

| reliable challenge metric | A | B | C | D | persistence | best G5 |
|---|---:|---:|---:|---:|---:|---|
| PushT agent XY | **31.413** | 34.247 | 33.358 | 34.007 | 49.940 | A |
| PushT block XY, standard spatial probe | 24.353 | 23.696 | 23.703 | **23.633** | 29.328 | D |
| PushT block XY, corrected L12 object crop | 23.114 | 23.545 | **22.541** | 22.863 | 28.996 | C |
| PushT rotation, corrected L12 object crop (deg) | 19.455 | 19.561 | 19.617 | **19.111** | **17.257** | none beats persistence |
| TwoRoom agent XY | 3.908 | 5.060 | **3.506** | 3.585 | 19.965 | C |
| Cube block XYZ | 0.0383 | 0.0430 | 0.0369 | **0.0335** | 0.0810 | D |
| Cube effector XYZ | 0.0253 | 0.0256 | 0.0256 | **0.0237** | 0.0749 | D |

The corrected post-norm probe passes the PushT challenge gate but not the
natural-distribution gate for both block XY and rotation. Standard global
rotation remains gate-failed and is diagnostic only. TwoRoom challenge and
natural agent metrics, Cube challenge block, and Cube challenge/natural
effector metrics pass their existing gates; Cube natural block does not.

For corrected object-local challenge errors, D-A episode-paired differences are
-0.183 [-1.013,+0.763] for block XY and -0.433 [-1.470,+0.553] degrees for
rotation. Both intervals include zero. More importantly, D block-XY correct
action error (22.863) is slightly worse than action shuffle (22.355), while C is
effectively tied (22.541 vs 22.535). D rotation separates correct from shuffle
(19.111 vs 21.341) but remains worse than persistence. Thus there is no robust
evidence that G5 learned the desired action-conditioned T-block interaction.

## 7. Hypothesis decisions and failure localization

1. **Spatial imbalance:** partially visible on Cube moving tokens, but rejected
   as a general explanation because B significantly harms PushT and TwoRoom and
   increases static-region error.
2. **Action insensitivity:** confirmed as a continuing symptom, but the proposed
   detached-negative ranking objective does not consistently fix it. Its benefit
   is environment-dependent and absent for reliable T-block translation.
3. **Interaction representation:** still the strongest unresolved bottleneck.
   A loss can reweight DINO patches but cannot by itself create stable object
   identity, localized action routing, or a persistent interaction state.
4. **Representation versus dynamics:** G4 already showed that corrected
   post-norm DINO features contain usable object pose signal. G5's continued
   rotation failure despite that reliable readout further localizes the main
   limitation to the dynamics/object-interaction mechanism, not the encoder
   target alone.
5. **Longer training:** A continuation improves macro feature error over H1, but
   corrected object-local PushT block XY degrades (20.91 to 23.11) and rotation
   remains below persistence. More updates to these objectives are not justified
   as the immediate next experiment.

## 8. Recommendation

Do **not** adopt arm B or C alone as the generalist default. Arm D is the most
promising of the four for broad feature prediction and Cube/TwoRoom challenge
metrics, but the evidence is too weak and environment-dependent to call it a
successful object-centric strategy. If a continuation checkpoint is needed for
diagnostic work, D is the least-bad candidate; it should not replace H1 as the
research baseline for PushT object interaction.

The next experiment should remain label-free and scalable but introduce a
minimal learned object/interaction mechanism rather than another loss-weight or
longer-budget sweep. Before training, it should pass these gates:

- selector/grouping uses only RGB/DINO and actions, with no masks or state;
- selected tokens are stable across time and do not collapse to the agent or
  background, measured only in evaluation;
- action intervention changes localized object predictions in the correct
  direction while preserving static regions;
- corrected post-norm PushT block XY and rotation are preregistered, with
  persistence and action shuffle as mandatory baselines;
- one neutral environment and one object-motion environment are used to detect
  negative transfer before any large run.

An explicit object-slot/token-routing dynamics module is now more justified
than extra training of A-D. A representation change should be minimal and
shared across environments; environment-specific masks, labels, rewards, or
contact definitions should remain evaluation-only.

## Artifacts and tests

- Shared cache/calibration: `shared_train_cache/`, `gradient_calibration.json`,
  `experiment_plan.json`
- Per-arm configs, 500-row metrics, checkpoints, status, and logs: the four arm
  directories plus launcher logs
- Common and corrected post-norm evaluation: `evaluation/`
- Episode summaries and paired bootstrap: `analysis/episode_metrics.csv` and
  `analysis/paired_bootstrap.json`
- Final checkpoints: `checkpoint_step2500.pt` in each arm directory

G5 targeted and regression tests passed (42/42), and the full world-model suite
passed (151/151, four existing warnings). Full repository collection still has
the known two optional PushT environment-test errors because `pygame` is not
installed; no dependency was silently added after the experiment. No G0-G4
artifact, dataset, or checkpoint was changed or deleted. No Git push, CEM,
additional seed, 2,000-step extension, or large architecture training occurred.
