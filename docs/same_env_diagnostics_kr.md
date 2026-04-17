# Same-Env Diagnostics

`same-env` 진단의 목적은 두 가지를 분리하는 것입니다.

1. `vision encoder`가 실제로 이미지를 쓰는지
2. `action formulation / execution`이 closed-loop에서 왜 무너지는지

지금 repo에서는 이걸 한 번에 보려고 [run_same_env_diagnostics.py](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/scripts/run_same_env_diagnostics.py)를 추가했습니다.

## 출력 파일

진단 결과 디렉토리에는 기본적으로 아래 파일이 생성됩니다.

- `summary.json`
- `image_usage.json`
- `initial_actions.json`
- `execute_steps_eval.json`
- `video_eval.json`
  - `--video-episodes > 0`일 때만 생성

## 핵심 지표

### 1. `offline_seen_env.rmse`

- same-env split에서 one-step supervised fit이 되는지
- 값이 낮은데 `online_seen_env.mean_reward`가 `0`이면
  - replay / supervised fit은 되지만
  - closed-loop rollout이 깨지는 상태입니다

### 2. `image_usage.zero_image_l2_delta`

- 이미지를 전부 0으로 만들었을 때 prediction이 얼마나 바뀌는지
- 거의 `0`이면 vision path가 죽어 있는 겁니다

### 3. `image_usage.zero_state_l2_delta`

- low-dim state를 0으로 만들었을 때 prediction이 얼마나 바뀌는지
- 이 값만 크고 `zero_image_l2_delta`가 작으면 state branch에 과도하게 의존하는 상태입니다

### 4. `image_usage.saturation_fraction`

- prediction이 `[-1, 1]` 경계에 얼마나 붙는지
- 이 값이 크면 action head saturation 가능성이 큽니다

### 5. `initial_action.std`

- episode마다 첫 action이 얼마나 달라지는지
- env가 바뀌거나 reset이 달라도 첫 action std가 매우 작으면
  - open-loop trajectory collapse
  - 또는 vision-conditioned branching 부족
로 해석할 수 있습니다

### 6. `execute_steps_eval`

- 동일 checkpoint에서 `execute_steps=1,2,4`를 바꿔가며 eval
- `execute_1`보다 `execute_2`가 확실히 좋으면
  - action horizon 자체보다
  - receding-horizon 재계획이 너무 잦아 생기는 진동 문제가 더 크다는 뜻입니다

## 자동 플래그

`summary.json`의 `diagnostic_flags`는 아래 의미를 가집니다.

- `vision_path_alive`
  - `zero_image_l2_delta.mean > 1e-3`
- `state_dominant`
  - state 제거 영향이 image 제거 영향보다 훨씬 큼
- `prediction_saturated`
  - saturation fraction이 높음
- `initial_action_collapsed`
  - 첫 action 분산이 비정상적으로 작음
- `execute_steps_sensitive`
  - `execute_steps`를 바꾸면 SR이 크게 바뀜
- `closed_loop_gap`
  - offline RMSE는 낮지만 online SR이 안 나옴

이 플래그는 최종 판정이 아니라 1차 triage 용도입니다.

## 권장 사용 순서

1. same-env `50 epoch` checkpoint를 만든다
2. `run_same_env_diagnostics.py`로 아래를 같이 본다
   - offline RMSE
   - online SR
   - image usage
   - initial action variance
   - execute_steps sweep
3. 필요하면 `--video-episodes 3`으로 GIF까지 저장한다

## 예시 명령

```bash
source scripts/common_env.sh
conda activate "$ELSA_ENV_NAME"

python scripts/run_same_env_diagnostics.py \
  --model-path /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/model_checkpoints/same_env_method_v2/slide_block_to_target/slide_keyframe4_dino_sameenv_e50_v2/env_000.pth \
  --task slide_block_to_target \
  --dataset-config-path experiments/slide_block_to_target_keyframe4_dinov3_sameenv.yaml \
  --env-id 0 \
  --eval-episodes 5 \
  --initial-action-episodes 5 \
  --execute-steps 1,2,4 \
  --device cuda:0 \
  --output-dir results/same_env_diagnostics/slide_keyframe4_dino_env0
```

## 지금 기준 해석 규칙

- `vision_path_alive = false`
  - encoder / branch wiring 문제부터 봐야 합니다
- `vision_path_alive = true`인데 `initial_action_collapsed = true`
  - 이미지는 보지만 policy head가 평균 trajectory로 collapse했을 가능성이 큽니다
- `execute_steps_sensitive = true`
  - action representation보다 execution semantics를 먼저 봐야 합니다
- `closed_loop_gap = true`
  - one-step BC 한계, chunking, keyframe, corrective behavior 부재 쪽을 봐야 합니다
