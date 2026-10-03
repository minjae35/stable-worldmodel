# Source-of-truth --- Document

## 1. Goal

Experiment 1은 heterogeneous 7-env에서 하나의 Generalist Visual World Model이 높은 prediction·planning 성능을 유지하는지 검증함. 핵심 비교는 **Dense Generalist B와 RD-WM C**이며, 작은 environment-specific dynamics residual이 성능 개선과 negative transfer 억제에 필요한지 확인함. Independent specialists A는 단독 학습 기준선이고, Pooled Generalist D는 spatial representation의 효과를 확인하는 비교 모델임. 본 문서는 구현의 source of truth이며, 변경은 명시적인 문서·config revision으로 기록함.

**Main suite:** PushT, TwoRoom, OGBench Cube, OGBench Scene, DMC Reacher, LP² PointMaze U-Maze, OGBench Visual AntMaze (`visual-antmaze-medium-navigate-v0`).

## 2. Frozen Shared Architecture

아래는 공통 기본 사양이며, 모델별 공유 범위와 D의 pooling 예외는 Section 3을 따름.

| Component | Frozen specification |
|---|---|
| Input | **RGB 224×224** |
| Encoder | Trainable ViT-Tiny, **width 192, depth 12, heads 3, MLP 768**; pretrained/frozen encoder 사용 안 함 |
| Patch / tokens | **Patch 14** → CLS를 제외한 **256 encoder tokens** → **2×2 spatial average pooling** → **64 dynamics tokens** |
| Visual projector | **192 → 768 → 192**, hidden LayerNorm + GELU, output normalization 없음 |
| Action block | **연속된 5 native actions**를 flatten함. 동일 action을 5번 반복하는 방식이 아님 |
| Action adapter | Env별 **`5 × native_action_dim → 192 → 192`**, GELU |
| Env embedding | Learned **192-D** vector; action embedding과 더해 모든 dynamics block의 AdaLN에 주입 |
| Dynamics | **6 blocks, width 192, heads 3, head_dim 64, FFN 768, dropout 0.1** |
| Position / mask | Spatial·temporal position embeddings. 같은 frame 내 전체 attention 허용, 미래 frame 차단 |
| History | **최대 3 frames**, observation stride **5 native steps** |
| Prediction head | **192 → 768 → 192**, hidden LayerNorm + GELU |
| Training objective | **2-step recursive latent prediction + SIGReg** |

- Encoder와 goal representation에는 env embedding을 넣지 않음.
- AdaLN modulation의 마지막 linear는 zero-init함.
- 두 번째 prediction은 첫 번째 **예측 latent**를 입력으로 사용함. Teacher forcing으로 대체하지 않음.
- Prediction·target encoder에 gradient를 전달하며, recursive latent를 detach하지 않음. EMA·stop-gradient 없음.
- Loss는 `0.5 × MSE(t+1) + 0.5 × MSE(t+2) + 0.09 × SIGReg`임. MSE는 batch·token·channel 평균임.
- SIGReg: **1024 projections, 17 knots**, 실제 encoded latents에 적용함. Spatial 모델은 batch마다 8개 위치를 선택하고, 위치·시점별로 **동일 env의 batch 축**에서 계산함. D는 pooled token에 적용함.
- Decoder·reward head·추가 auxiliary loss는 사용하지 않음.

## 3. Model Variants

| Variant | 정확한 차이 |
|---|---|
| **A. Independent Specialists** | Env별 독립 encoder·action interface·dynamics를 학습함. 64 spatial tokens를 사용하며 residual adapter 없음 |
| **B. Dense Generalist** | Shared encoder + env-specific action interface + env embedding + fully shared dynamics. Residual adapter 없음 |
| **C. RD-WM** | B와 동일하되 **block 5, 6 출력에만 env-specific residual adapter** 추가 |
| **D. Pooled Generalist** | B와 동일한 encoder·2×2 pooling·visual projector를 사용한 뒤, 64 spatial latent tokens를 global mean pooling하여 **1 token**으로 변환함. CLS 사용 안 함. Residual adapter 없음 |

C의 adapter는 다음과 같음.

`h_out = h + Linear(32,192)(GELU(Linear(192,32)(LN(h))))`

- Env별·block별 독립 parameters를 사용함.
- Up projection `Linear(32,192)`의 **weight와 bias를 0으로 초기화**함.
- 동일 shared initialization에서 B와 C의 초기 출력은 같아야 함.

**B와 C의 유일한 핵심 architecture 차이는 dynamics residual specialization임.** Encoder, action interface, env conditioning, loss, data, training schedule, planner를 함께 바꾸지 않음.

## 4. Data / Training Contract

- 검증된 **7개 Lance source를 유지**하고 URI/revision·column mapping·action alignment를 manifest에 기록함.
- **Env-homogeneous batch, batch size 32**를 사용함. Generalist는 7-step round마다 모든 env를 한 번씩 사용하고 순서를 shuffle함. Batch마다 optimizer update를 수행함.
- 모델의 데이터 입력은 **RGB + action**임. State metadata는 simulator restore·goal 설정·evaluation에만 사용함.
- Scene/AntMaze의 **64×64 원본을 input pipeline에서 224×224로 resize**함. Train·live observation·goal에 동일한 resize와 ImageNet normalization을 적용함.
- Full-history sample은 **5 observations + 연결하는 4 action blocks**임. Shape는 `pixels=[B,5,3,224,224]`, `actions=[B,4,5,d_env]`임.
- History 길이는 batch별 **3/2/1을 80%/10%/10%**로 사용함. Target 시점은 유지하고 앞쪽 context만 제거함.
- Split은 **episode 단위**이며 overlapping clips를 split 사이에 섞지 않음. Action mean/std는 train split에서만 계산하고 std 하한은 `1e-3`임.
- Action 시점 정렬과 terminal boundary를 검증함. LP²는 이미 보정된 source에 중복 shift하지 않음.
- **AdamW, lr `5e-5`, weight decay `1e-3`**, bias·normalization parameters는 weight decay 제외. **1% warmup + step-wise cosine decay** 사용.
- **BF16 autocast**, loss·SIGReg는 FP32, **gradient clip 1.0**. Random crop·flip·강한 color jitter 없음.
- Full v1 budget: **A는 env당 10,000 updates**, **B/C/D는 각각 70,000 updates**로 env별 sample 노출량을 맞춤. Training seeds는 **3개**임.

## 5. Evaluation

- **Primary metric: closed-loop CEM planning success.** Env별 success, macro average, worst-env success를 보고함.
- Prediction은 secondary임. Rollout **1/2/5/10 steps**의 error와 persistence 대비 결과를 기록하며, 서로 다른 learned latent의 raw MSE만으로 모델 우열을 판단하지 않음.
- 핵심 비교는 **B vs C**임. 같은 shared initialization, batch 순서, split, start/goal pairs, CEM budget을 사용함.
- CEM: **300 candidates, 30 iterations, 30 elites, horizon 5 blocks, action block 5, receding horizon 1 block, history max 3, solver env batch 1**.
- Cost는 **마지막 predicted visual latent와 goal latent의 mean squared distance**임. Action/env embedding은 cost에 포함하지 않음.
- CEM의 native-action bounds를 model rollout·elite update·실제 execution에 동일하게 적용함. Action 정규화는 모델 내부에서 한 번만 수행함.
- Main protocol: held-out trajectory의 **25 native steps 이후 goal**, execution budget **50 native steps**, env당 고정 test **100 pairs**. 시작부터 성공한 pair는 제외함.
- Validation macro planning success로 **모델당 checkpoint 하나**를 선택함. Test 결과로 선택하거나 env마다 다른 checkpoint를 사용하지 않음.
- Specialist 대비 env별 success 차이와 total/active parameters, GPU-hours, peak VRAM, planning latency를 함께 보고함. C의 우위를 specialization 효과로 주장하려면 기존에 정한 **parameter-matched shared-capacity control**을 추가함.

## 6. Experiment Order

**Phase 1 gate ✅ COMPLETE → Dense B → CEM → RD-WM C → 2-env smoke → 7-env pilot → A/B/C/D full experiment**

- **Phase 1 gate:** 7-env restore·action replay·goal/success 검증. Scene button state, LP² alignment, AntMaze camera 확인.
    - 7-env restore, action replay, goal/success 검증 통과.
- **2-env smoke:** PushT + Cube, B/C 각각 총 **1,000 updates**. Gradient·causal mask·recursive prediction·CEM 연결 확인.
- **7-env pilot:** B/C 각각 총 **7,000 updates**, env당 validation planning **20 pairs**. Collapse·sampling·planning 동작 확인.
- **Full:** 먼저 1 seed의 전체 matrix를 완성하고, protocol 고정 후 나머지 2 seeds 실행함.

## 7. Frozen vs Not Frozen

**Frozen:**

- 현재 7-env main suite와 RGB/action-only training.
- Section 2의 architecture·token configuration·conditioning·loss.
- A/B/C/D의 정의, 특히 B/C의 residual specialization 차이.
- Batch 32, env-balanced sampling, episode split, train-only normalization.
- 현재 v1 training/evaluation 수치와 동일 데이터 노출량·동일 planning budget 비교.
- 구현 중 변경이 필요하면 먼저 문서·config revision을 기록하고 비교 모델에 일관되게 반영함.

**Not Frozen:**

- 최종 채택 모델이 Dense B인지 RD-WM C인지 여부.
- Pilot에서 측정한 병목에 따른 worker·prefetch·cache 설정.
- Pilot evidence에 근거한 후속 버전의 capacity·loss weight·training/planning budget 조정. 기존 v1 결과와 구분하여 기록함.
- Main 이후의 low-data adaptation, positive-transfer 분석, environment scaling 범위.