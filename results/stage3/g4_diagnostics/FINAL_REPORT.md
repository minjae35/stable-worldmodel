# Stable-WM Stage 3 G4 final report

## Outcome

G4 completed the representation diagnostic, all pre-training checks, one
pre-registered H3 + fixed sinusoidal temporal encoding run (2,000 updates), and
the common Development evaluation. The experiment does **not** support adopting
explicit sinusoidal temporal encoding as the next default: at step 2,000 it is
essentially neutral on PushT, significantly worse on TwoRoom, and significantly
better only on Cube relative to H3. It also does not solve PushT T-block
rotation.

Frozen DINOv2 does retain useful T-block rotation information. The dominant
representation-side failure is the global/spatial readout, not absence of the
signal in the encoder. The stronger evidence points next to object-centric
spatial/temporal dynamics and an interaction-sensitive objective/evaluation,
not an encoder replacement or another positional-encoding sweep.

## Important evaluation correction

`Dinov2Model.hidden_states[12]` is before the encoder's final LayerNorm, while
`Dinov2Model.last_hidden_state` and the SpatialDynamics target are after that
LayerNorm. The first object-local evaluation mixed these spaces. Those artifacts
were preserved and marked `SUPERSEDED`; no checkpoint or original result was
deleted. The authoritative representation diagnostic is
`representation_postnorm`, and the authoritative common object evaluation is
`object_readout_evaluations_postnorm`.

## G4-A representation diagnostic

- Split: 1,024/256/256 windows and 256/64/64 episodes for train/validation/test;
  episodes are disjoint. Ridge alpha and readout are selected only on validation.
- Directed asymmetric T-block angle is represented as sin/cos(theta), evaluated
  with 2-pi circular error. Object centroid/state are evaluation-only.
- Final layer means the exact post-final-LayerNorm token target used by dynamics.
  Intermediate layers are native block outputs.

| Readout | absolute rotation MAE | delta MAE | direction accuracy | block XY L2 |
|---|---:|---:|---:|---:|
| L12 CLS | 17.55 deg | 14.45 deg | 61.8% | 30.96 |
| L12 pool-4 spatial | 11.39 deg | 9.15 deg | 65.7% | 12.44 |
| L12 object crop-4 | 4.34 deg | 3.89 deg | 80.4% | 7.52 |
| L6 object crop-4 | **2.55 deg** | **2.12 deg** | **90.2%** | **5.64** |

L6 object-local is validation-selected best overall; L12 object-local is the
validation-selected readout that can be applied directly to dynamics output.
For L6, delta MAE is 1.61/3.66/3.62 deg for small/medium/large rotations;
direction accuracy is 73.7%/100%/100%. Approach-into-contact and
contact-at-current delta MAEs are 1.49 and 2.17 deg respectively.
The small MLP on L6 object-local is worse than ridge (6.64 deg test MAE), so the
linear result is not explained by insufficient probe capacity. A direct paired
feature-difference probe is also worse (5.10 deg, 52.9% direction) than
differencing absolute predictions. RGB orientation gives 4.72 deg absolute MAE;
for medium/large motion its delta errors are 1.47/1.29 deg. The DINO signal is
there, but spatial localization matters strongly and L6 preserves it better than
the final layer.

The RGB contact proxy leaves only one no-contact test window; contact-stratified
claims therefore remain limited. Object-local cropping uses an oracle
RGB-derived centroid and is strictly an evaluation diagnostic, not a training
input or a deployable readout.

## G4-B controlled training

- H3, uniform label-free sampling, MSE, fresh seed 3072.
- Batch 16/environment (48/update), AdamW 1e-3, weight decay 1e-4,
  gradient clip 1.0, frameskip 5, 2,000 updates.
- Fixed sinusoidal positions `[0,1,2]`, scale 1.0, added once after token
  projection plus action/environment embeddings and before the factorized block
  stack.
- No trainable parameters added: both H3 and G4 have 6,794,368 parameters.
- Initial model hash `5d5da87d...54dfa2`; all three schedule hashes and the
  optimizer setup match G3 H3. The encoder stayed frozen and training inputs were
  only episode index, RGB pixels, and actions.
- Checkpoints were saved at 500/1,000/1,500/2,000. No NaN, Inf, or OOM occurred.
- Training took 363.15 s of measured step time (6.05 min), averaging 0.1816
  s/update and 271.8 samples/s; peak allocated VRAM was 5.917 GiB.

The causal mask blocks future frames but, without positional encoding, attention
over already-visible keys does not explicitly tag their order. The fixed encoding
adds that tag without changing parameter count. Alignment/leakage, shape/index,
forward/backward, frozen encoder, initialization/schedule equality, historical
actions, checkpoint/RNG resume, regression, and GPU smoke checks passed before
the full run.

## Feature prediction results

Macro values average PushT, TwoRoom, and Cube equally.

| step | H1 macro MSE | H3 macro MSE | G4 macro MSE | G4 PushT | G4 TwoRoom | G4 Cube |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.405443 | 0.402975 | **0.400574** | 0.291708 | 0.227185 | 0.682828 |
| 1,000 | **0.334224** | 0.333331 | 0.336090 | 0.245729 | 0.167954 | 0.594586 |
| 1,500 | **0.299553** | 0.309241 | 0.305108 | 0.229441 | 0.136389 | 0.549494 |
| 2,000 | **0.285417** | 0.296949 | 0.297871 | 0.223021 | 0.132211 | 0.538383 |

At step 2,000 the macro MAEs are H1 0.356337, H3 0.359428, and G4 0.361532.
G4 versus H3 episode-paired MSE effects (negative favors G4) are:

| step | PushT delta (95% CI) | TwoRoom delta (95% CI) | Cube delta (95% CI) |
|---:|---:|---:|---:|
| 500 | -0.000981 [-0.001906,-0.000069] | -0.006676 [-0.008883,-0.004508] | +0.000619 [-0.001653,+0.003001] |
| 1,000 | +0.000191 [-0.002032,+0.002416] | +0.008018 [+0.004724,+0.011186] | -0.000026 [-0.003704,+0.003683] |
| 1,500 | -0.002512 [-0.006105,+0.000821] | -0.001392 [-0.004668,+0.001748] | -0.008274 [-0.012970,-0.003932] |
| 2,000 | +0.001453 [-0.001294,+0.004258] | +0.005694 [+0.000824,+0.010400] | -0.004997 [-0.009184,-0.000819] |

Thus the effect changes with training step. The final PushT result is
inconclusive (+0.66%), TwoRoom is worse (+4.58%), and Cube is better (-0.92%).
This is one seed and must not be treated as a reproducible architecture win.

## State and interaction diagnostics

The corrected L12 object-local probe passes the challenge rotation gate:
true-DINO absolute/delta errors are 4.94/6.88 deg for mean true motion 17.70 deg
(margin 2.57). Natural rotation and translation have motions smaller than probe
error and remain `diagnostic_only`.

| step | H1 rot MAE | H3 rot MAE | G4 rot MAE | persistence | G4 direction |
|---:|---:|---:|---:|---:|---:|
| 500 | 23.36 | 24.21 | 22.63 | 17.26 | 72.7% |
| 1,000 | 21.94 | 24.39 | 22.39 | 17.26 | 68.2% |
| 1,500 | 19.53 | 23.44 | 20.47 | 17.26 | 65.9% |
| 2,000 | **18.93** | 21.92 | 23.18 | **17.26** | 65.9% |

No model beats persistence on reliable PushT rotation at any reported step.
At step 2,000 G4 is also worse than action shuffle (19.97 deg), which is direct
evidence that the learned object rotation is not reliably action-dependent.

For reliable object-local PushT block XY challenge at step 2,000, correct-action
MAEs are H1/H3/G4 = 20.91/23.58/23.61 versus persistence 29.00. Translation is
learned beyond persistence, but correct action is almost tied with action shuffle
for G4 (23.39), so action specificity remains weak.

The existing validated challenge probes show G4 correct-action MAE at step 2,000:
PushT agent 35.01 (persistence 49.94), TwoRoom agent 5.14 (19.96), Cube block
0.03696 (0.08104), and Cube effector 0.02570 (0.07493). These beat persistence,
but relative to H3 G4 worsens PushT agent and TwoRoom, improves Cube block, and
is essentially neutral on Cube effector. Natural small-motion state metrics that
fail their probe gates remain diagnostic-only.

## History and order sensitivity

At step 2,000, swapping the two historical pairs changes MSE relative to the
unaltered input by:

| model | PushT | TwoRoom | Cube |
|---|---:|---:|---:|
| H3 | +0.32% | +1.90% | +0.70% |
| G4 | -0.05% | +2.11% | +1.78% |

Temporal encoding increases order sensitivity meaningfully only on Cube and
slightly on TwoRoom; PushT remains insensitive. Historical-action shuffle
penalties are H3 3.12/9.21/5.94% and G4 4.13/7.06/6.81% for
PushT/TwoRoom/Cube. G4 uses history and historical actions, but does not turn
that use into a general performance gain.

## Validation and preserved artifacts

- Pre-training world-model suite: 138 passed, 4 existing warnings.
- Final world-model suite: 138 passed, 4 warnings.
- A full repository collection attempt reports two optional environment-test
  collection errors because `pygame` is not installed; this is separate from
  the G4 tests and was not hidden by installing a new dependency post hoc.
- Syntax compilation and `git diff --check` pass.
- Corrected post-norm object evaluation: 12/12 jobs completed in 98.90 s.
- Initial model, full per-environment schedules, parameter count, optimizer,
  action normalizer, split, seed, and target schedule match H3; fairness PASS.
- Resume smoke reproduces objectives and state restoration; independent CUDA
  processes differ by at most 1.91e-6 in final parameters, recorded as numerical
  tolerance rather than bitwise identity.
- No G0/G1/G2/G3 result, dataset, or checkpoint was overwritten. No Git push,
  extra seed, CEM, MoE, gradient surgery, or other architecture training ran.

## Research judgement

1. DINOv2 contains enough PushT rotation signal for the current research stage.
   Encoder replacement is not the first priority.
2. Intermediate L6 object-local features preserve rotation best, but even the
   exact post-norm L12 dynamics target supports a much better readout than the
   old global probe. The earlier evaluation was primarily probe/readout-limited.
3. Fixed temporal encoding does not generally improve H3. It helps Cube, hurts
   TwoRoom, and is neutral/inconclusive on PushT at the final checkpoint.
4. PushT rotation still fails against persistence despite a now-reliable probe.
   This localizes the main limitation after the encoder: the shared dynamics and
   its pixel-averaged MSE objective do not sufficiently prioritize localized,
   action-conditioned object interaction.
5. The next architecture direction should be object-centric but label-free:
   learned token selection/slot-style grouping or motion-aware token routing,
   coupled with a temporal transition mechanism that models localized
   action-object effects. Before training it, lock an action-shuffle-sensitive
   object metric and verify the learned selector without masks/state labels.
6. Objective work is the second priority: foreground/motion-aware or contrastive
   token losses derived only from RGB/DINO changes, with gradient-scale controls.
   Another encoder, history-length, or positional-encoding sweep is lower value
   until the localized interaction bottleneck is addressed.
