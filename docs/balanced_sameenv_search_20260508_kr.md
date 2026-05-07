# Balanced Same-Env Search Plan - 2026-05-08

목표는 `close_box`, `slide_block_to_target`, `insert_onto_square_peg`, `scoop_with_spatula` 4개 task가 같은 환경에서 고르게 성공하는 action + vision representation을 찾는 것이다. Slide만 높은 결과는 더 이상 충분하지 않다.

## 현재 막힌 지점

- 2026-05-07 4GPU 큐의 여러 run은 50 epoch 학습까지 끝났지만 live eval에서 `colosseum` package collision 때문에 실패했다.
- 원인은 pip의 CSS `colosseum`이 먼저 import되고, 실제 필요한 `/home/cvlab-dgx/siwon/robot-colosseum`이 `PYTHONPATH`에 없던 것이다.
- `elsa_learning_agent/utils.py`와 `scripts/prepare_live_eval_env.sh`에서 robot-colosseum을 우선 import하도록 수정했다.
- 우선 `recovered_live_eval_20260508` tmux 세션으로 실패했던 checkpoint들을 재평가한다.

## FLAME metadata validity rule

학습 피처와 auxiliary supervision은 FLAME/RLBench observation 또는 demonstration label에서 얻을 수 있는 것만 유효하다.

허용:

- `front_rgb`, optional previous `front_rgb` for temporal pair
- proprio low-dimensional state
- `joint_positions`, `joint_velocities`
- `gripper_open`
- `gripper_pose[:3]` as EE auxiliary target
- `obs.misc.front_camera_intrinsics`
- `obs.misc.front_camera_extrinsics`
- `obs.misc.front_camera_near`, `obs.misc.front_camera_far`
- expert action labels derived from the same trajectory
- gripper transition mask derived only from expert `gripper_open` changes

금지:

- rollout reward or success label as a training feature
- eval/test environment privileged state
- hard-coded object pose/state unless it is already present in the FLAME observation stream used by the policy
- replay-oracle target during learned rollout
- task-specific manual success detector as input

정리하면 camera-aware VolumeDP는 유효하다. 단, camera K/T는 client-local metadata로만 사용하고 서버에 raw metadata를 보내는 설정으로 주장하지 않는다. `include_camera_in_state=true`는 same-env에서 이미 성능을 해쳤으므로 기본 search에서는 쓰지 않는다.

## 현재 회수 중인 결과

tmux:

```bash
tmux attach -t recovered_live_eval_20260508
```

logs:

```bash
tail -f /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/recovered_live_eval_20260508/gpu0.log
tail -f /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/recovered_live_eval_20260508/gpu1.log
tail -f /mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/recovered_live_eval_20260508/gpu3.log
```

outputs:

```bash
/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/recovered_live_eval_20260508
```

회수 대상:

| Task | Checkpoint | Modes |
| --- | --- | --- |
| slide | `slide_jprel_w2_direct_grid16_eeaux_e50_s0` | threshold |
| close | `close_jprel_w4_jvservo_grid16_eeaux_e50_s0` | threshold, hysteresis |
| insert | `insert_jprel_w4_jvservo_grid16_eeaux_e50_s0` | threshold, hysteresis |
| scoop | `scoop_jprel_w4_jvservo_grid16_eeaux_e50_s0` | threshold, hysteresis |
| insert | `insert_jpabs_w2_grid16_e50_s0` | threshold |
| scoop | `scoop_jpabs_w2_grid16_e50_s0` | threshold |

## 다음 학습 큐

현재 active queue와 recovered eval이 끝나면 `balanced_sameenv_search_20260508`을 시작한다.

핵심 설정:

- vision: `volumedp_full_dinov3_depth`
- metadata: FLAME camera K/T, near/far, EE position only through observation context
- action: `jprel_w4_direct` and `jprel_w4_jvservo`
- loss: diffusion arm + separate gripper BCE + transition-weighted gripper BCE
- gripper transition window: `8`
- gripper transition weight: `6.0`
- epochs: default `100`
- eval: same-env `env_000`, 20 episodes

우선순위:

| Priority | Task | Config | Reason |
| --- | --- | --- | --- |
| 1 | close | `jprel_w4_direct_gripw6xk8` | close는 gripper timing 영향이 크고 direct가 slide에서 가장 강했다. |
| 2 | insert | `jprel_w4_jvservo_gripw6xk8` | replay ceiling상 JP-to-JV servo가 가능성이 높다. |
| 3 | scoop | `jprel_w4_jvservo_gripw6xk8` | scoop도 replay ceiling상 servo가 더 안전한 후보다. |
| 4 | slide | `jprel_w4_direct_gripw6xk8` | 기존 최고 slide 0.85를 유지하는 regression guard다. |

시작 명령:

```bash
bash scripts/start_balanced_sameenv_search_20260508_tmux.sh
```

## 판단 기준

- `min(SR over 4 tasks)`를 1차 목표로 본다.
- slide가 높고 나머지가 0인 조합은 탈락이다.
- close/insert/scoop 중 하나라도 `0.10` 이상이면 해당 action contract를 확장한다.
- hysteresis eval이 threshold보다 좋으면 이후 rollout 기본값 후보로 올린다.
- e50에서 loss가 계속 내려가는데 SR이 낮으면 e100/e150으로 늘린다.
- e100에서도 non-slide가 0이면 action보다 gripper event head 또는 task phase head가 필요하다.

