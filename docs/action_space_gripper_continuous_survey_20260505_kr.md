# Action Space / Gripper Survey (2026-05-05)

## 요약

다른 repo들의 공통 패턴은 `arm continuous + gripper separate command`다. Keyframe 계열(PerAct/RVT/SAM2Act)은 EE pose를 keyframe으로 예측하더라도 gripper는 별도 binary class로 둔다. 연속 행동 계열(ACT, Diffusion Policy/robomimic, DP3)은 action vector 마지막 1차원을 gripper로 붙이지만, 변환/손실에서 gripper를 arm pose와 분리해서 처리한다.

우리 현재 구현도 diffusion arm과 gripper BCE head를 분리해 두었지만, same-env 실패 task에서는 gripper 전환이 episode당 1-2회로 희소하다. 따라서 단순 chunk BCE만으로는 "언제 닫고/열지"가 약하게 학습될 가능성이 크다.

## 코드 기준 확인

### RLBench / Colosseum 계열

- `JointVelocity`: 7D arm joint velocity를 1 step 적용한 뒤 zero velocity로 reset한다. 로컬 코드: `/home/cvlab-dgx/siwon/RLBench-stepjam/rlbench/action_modes/arm_action_modes.py:59`.
- `JointPosition`: absolute 또는 delta target joint position이다. delta mode는 current joint positions에 action을 더한다. 로컬 코드: `/home/cvlab-dgx/siwon/RLBench-stepjam/rlbench/action_modes/arm_action_modes.py:88`.
- `Discrete` gripper: action shape `(1,)`, `>0.5`는 open, 이하는 close다. close 시 grasp object를 attach하고 open 시 release한다. 로컬 코드: `/home/cvlab-dgx/siwon/RLBench-stepjam/rlbench/action_modes/gripper_action_modes.py:40`.
- 결론: RLBench 연속 arm action을 쓰더라도 gripper는 보통 별도 1D discrete command다.

### PerAct / RVT / SAM2Act

- PerAct replay buffer action shape는 `(8,)`이고, 내부 target은 `trans_action_indicies`, `rot_grip_action_indicies`, `ignore_collisions`, `gripper_pose`로 나뉜다. 로컬 코드: `/home/cvlab-dgx/siwon/peract/agents/peract_bc/launch_utils.py:70`.
- PerAct `rot_grip_action_indicies`는 3개 rotation bin + 1개 gripper class다. training에서 gripper one-hot을 따로 만든다. 로컬 코드: `/home/cvlab-dgx/siwon/peract/agents/peract_bc/qattention_peract_bc_agent.py:367`.
- RVT도 gripper one-hot과 collision one-hot을 따로 둔다. 로컬 코드: `/home/cvlab-dgx/siwon/RVT/rvt/models/rvt_agent.py:412`.
- SAM2Act replay action shape는 `(8,) = 3 translation + 4 quat + 1 gripper_open`이라고 명시되어 있다. 로컬 코드: `/home/cvlab-dgx/siwon/sam2act/sam2act/utils/dataset.py:140`.
- 결론: keyframe 계열은 action이 high-level이어도 gripper를 arm/pose regression에 섞지 않는다.

### ACT / ALOHA

- ACT sim action space는 `[left_arm_qpos(6), left_gripper(1), right_arm_qpos(6), right_gripper(1)]`다. gripper는 normalized continuous position `(0 close, 1 open)`이다. 로컬 코드: `/home/cvlab-dgx/siwon/visionencoder/act/sim_env.py:20`.
- `before_step`에서 arm qpos와 gripper command를 분리하고, gripper 1D command를 실제 양 finger joint command로 unnormalize한다. 로컬 코드: `/home/cvlab-dgx/siwon/visionencoder/act/sim_env.py:58`.
- 결론: continuous action chunk에서도 gripper는 action vector에 들어가지만 의미는 arm joint와 다른 normalized gripper position이다.

### Diffusion Policy / robomimic / DP3

- Diffusion Policy의 robomimic action conversion은 delta arm action을 absolute pose로 바꾸면서 gripper action은 그대로 보존한다. 로컬 코드: `/home/cvlab-dgx/siwon/visionencoder/diffusion_policy/diffusion_policy/common/robomimic_util.py:52`.
- UMI/MuJoCo dataset 쪽은 `T, 8 = x,y,z,qx,qy,qz,qw,gripper_width` 형태다. 로컬 코드: `/home/cvlab-dgx/siwon/visionencoder/diffusion_policy/diffusion_policy/dataset/mujoco_image_dataset.py:63`.
- DP3 계열도 7D/8D action을 구분하고, gripper/progress를 별도 처리한다. gripper는 BCE loss로 둔다. 로컬 코드: `/home/cvlab-dgx/siwon/object_centric_diffusion/diffusion_policy_3d/policy/simple_dp3.py:163`, `/home/cvlab-dgx/siwon/object_centric_diffusion/diffusion_policy_3d/policy/simple_dp3.py:578`.
- 결론: 연속 diffusion policy에서도 gripper는 마지막 scalar지만, pose/rotation MSE와 같은 방식으로 취급하지 않는 구현이 많다.

## 우리 데이터에서 gripper 전환 빈도

`/mnt/raid0/siwon/data/ELSA-Robotics-Challenge/datasets/training`의 env 0/1, 200 episodes 기준:

| Task | transitions / ep | open fraction | median transition fraction |
| --- | ---: | ---: | ---: |
| slide_block_to_target | 1.00 | 0.751 | 0.760 |
| close_box | 2.00 | 0.700 | 0.699 |
| insert_onto_square_peg | 1.92 | 0.473 | 0.819 |
| scoop_with_spatula | 1.00 | 0.695 | 0.700 |

해석:

- `slide`가 잘 되는 이유가 "gripper가 필요 없다"는 뜻은 아니다. 실제 데이터는 마지막 25% 지점쯤 gripper가 close로 바뀐다. 다만 slide는 성공 조건이 arm path/pose에 더 민감하고 gripper 타이밍 실패 비용이 상대적으로 낮다.
- `close`, `insert`, `scoop`은 arm path가 맞아도 close/open 타이밍이 어긋나면 실패한다. 특히 insert는 두 번째 전환이 episode 끝 근처라 chunk 학습에서 희석되기 쉽다.

## 추천 action ablation

1. `jprel/w4 + discrete gripper head + transition-balanced BCE`

- 현재 best slide인 relative joint direct를 유지한다.
- gripper BCE에 transition window 가중치를 둔다. 예: gripper change 전후 `+-8` frame의 gripper loss weight를 4-8배.
- 이유: 다른 repo 패턴과 맞고, 현재 구현 변경량이 작다.

2. `jprel/w4 + gripper phase head`

- gripper scalar를 직접 예측하는 대신 `{hold_open, close, hold_closed, open}` 4-class로 예측한다.
- execution에서는 current gripper state와 class를 조합해 binary command를 만든다.
- 이유: close/open은 sparse event라 state 값보다 "전환 event"가 더 중요한 task가 있다.

3. `continuous arm + gripper hysteresis`

- sigmoid threshold를 단순 0.5로 자르지 말고 current gripper state 기반 hysteresis를 둔다. 예: open->close는 `p_close > 0.65`, close->open은 `p_open > 0.65`.
- 이유: gripper가 한두 frame 흔들리면 grasp attach/release가 망가진다. RLBench Discrete gripper는 전환 시 실제 grasp/release side effect가 크다.

4. `w2/w4 mixed chunk`

- arm은 `w4`를 쓰되 gripper head는 `w2` 또는 per-step event head를 추가한다.
- 이유: long chunk diffusion은 arm smoothing에는 좋지만, gripper 전환은 frame-level 타이밍이 더 중요하다.

## 실험 우선순위

1. `jprel_w4_direct_grid16_eeaux + transition_weighted_gripper_bce`
2. `jprel_w4_direct_grid16_eeaux + gripper_event_4class`
3. `jprel_w4_jvservo_grid16_eeaux + transition_weighted_gripper_bce`
4. `jprel_w2_direct_grid16_eeaux + transition_weighted_gripper_bce`

지금 바로 붙이기에는 1번이 가장 안전하다. action space 자체는 유지하고 gripper supervision만 강화하기 때문에, 이미 slide 0.85가 나온 relative joint direct 성능을 덜 망가뜨릴 가능성이 높다.

## 외부 source links

- RLBench: https://github.com/stepjam/RLBench
- PerAct: https://github.com/peract/peract
- RVT: https://github.com/NVlabs/RVT
- SAM2Act: https://github.com/sam2act/SAM2Act
- ACT: https://github.com/tonyzhaozh/act
- Diffusion Policy: https://github.com/real-stanford/diffusion_policy
