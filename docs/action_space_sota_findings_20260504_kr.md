# Action Space SOTA 조사 메모

작성일: 2026-05-04

목표:
- `slide_block_to_target`, `close_box`, `insert_onto_square_peg`, `scoop_with_spatula` 4개 task의 same-env 성공률을 먼저 올린다.
- FL 확장 전에 action formulation 병목을 줄인다.

## 조사 요약

RLBench 공식 예제는 `JointVelocity + Discrete gripper`를 기본 예제로 많이 쓴다.
- RLBench GitHub 예제: `MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())`
- 링크: https://github.com/stepjam/RLBench

하지만 RLBench SOTA 계열은 대체로 dense joint velocity를 직접 예측하지 않는다.
- PerAct는 RGB-D voxel observation에서 다음 keyframe의 translation, rotation, gripper state, collision action을 분류하고 motion planner로 실행한다.
- PerAct는 keyframe을 `joint velocity near zero` 및 gripper state 조건으로 뽑고, dense continuous action 직접 예측이 비효율적/노이즈가 크다고 설명한다.
- 링크: https://arxiv.org/abs/2209.05451

RVT/RVT-2도 같은 큰 방향이다.
- RVT-2는 다음 key-frame pose를 예측하고 planner가 trajectory를 생성하는 구조다.
- RVT-2는 RLBench 평균 성공률을 65%에서 82%로 올렸고, peg/plug insertion 같은 precision task를 강조한다.
- 링크: https://arxiv.org/abs/2406.08545

더 최신 RLBench 계열도 저수준 JV 직접 회귀보다는 3D/keyframe action interface를 유지한다.
- SAM2Act는 multi-view robotic transformer에 foundation-model visual representation과 memory를 붙여 RLBench 18-task 평균 86.8%를 보고했다.
- 링크: https://arxiv.org/abs/2501.18564

ACT/ALOHA 계열은 motion planner/keyframe 대신 joint target action chunk를 쓴다.
- ACT는 현재 관측에서 다음 `k` step의 absolute joint position sequence를 예측한다.
- 논문 내 ablation에서 delta joint position보다 target joint position이 나았다고 보고한다.
- 링크: https://openreview.net/pdf?id=e8Eu1lqLaf

UMI/OpenPI/LeRobot 계열은 relative trajectory를 중요하게 본다.
- UMI는 action chunk 안의 EE pose들을 같은 inference 시작 pose 기준 relative trajectory로 표현한다.
- delta action은 step마다 이전 action 기준이라 error가 누적된다고 설명한다.
- 링크: https://umi-gripper.github.io/umi.pdf
- LeRobot action representation 문서도 relative action을 `absolute - current_state`로 두고, gripper는 relative에서 제외하는 구성을 권한다.
- 링크: https://huggingface.co/docs/lerobot/action_representations

ManiSkill 같은 조작 시뮬레이터는 task-space EE delta pose controller를 명시적으로 지원한다.
- EE delta pose action을 IK로 joint target position으로 바꾸고 PD controller로 따라간다.
- 링크: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html

## 이 repo에 바로 적용한 결론

1. 기존 RLBench 공식 JV direct는 baseline으로 유지한다.
2. SOTA 방향의 핵심은 저수준 velocity 회귀보다 `target pose/target position`이다.
3. 지금 당장 구현 비용이 낮은 후보는 `relative joint target chunk`다.
4. EE keyframe pose + planner는 더 강한 방향이지만, dataset/eval action target을 EE pose로 바꾸고 collision/planning 실패 처리까지 봐야 해서 후속 작업으로 둔다.

## 새 실험

추가 config:
- `experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_direct_grid16_eeaux.yaml`

핵심 설정:
- action target: `joint_position_relative`
- chunk: 4
- execute: 2
- gripper: absolute binary/open value 유지
- execution: relative joint target을 inference 시작 관측 기준 absolute joint target으로 변환한 뒤 `JointPosition(absolute)` 실행
- action bounds: joint delta `[-0.25, 0.25]`, gripper `[0, 1]`

env0 데이터 통계:
- 4-step relative joint delta max는 4개 task 모두 약 `0.20 rad` 이하.
- 따라서 `[-0.25, 0.25]` bounds는 clipping 없이 normalization을 촘촘하게 만든다.

실행 스크립트:
- `scripts/start_relative_action_4task_queue_tmux.sh`

큐 동작:
- 현재 overnight/recommended/paperfaithful 학습이 끝날 때까지 대기한다.
- 이후 GPU 0~3에 4개 task same-env e50을 병렬로 실행한다.

## 추가 action ablation

추가 config:
- `experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w4_jvservo_grid16_eeaux.yaml`
- `experiments/sameenv_volumedp_full_dinov3_depth_lora8_jprel_w2_direct_grid16_eeaux.yaml`

추가 실행 스크립트:
- `scripts/start_action_ablation_queue_tmux.sh`

의도:
- `jprel_w4_jvservo`: target은 그대로 두고 실행 interface만 `JointPosition direct`에서 benchmark-compatible `JointVelocity servo`로 바꾼다. 성공률 차이가 나면 target formulation보다 low-level execution이 병목이다.
- `jprel_w2_direct`: 같은 relative target이지만 chunk horizon을 4에서 2로 줄이고 action bounds를 `[-0.15, 0.15]`로 좁힌다. open-loop drift가 문제인지, longer chunk supervision이 이득인지 확인한다.

env0 데이터 통계:
- 2-step relative joint delta max는 4개 task global max 약 `0.10 rad`라서 `[-0.15, 0.15]` bounds가 안전하다.
- 4-step relative joint delta max는 4개 task global max 약 `0.20 rad`라서 `[-0.25, 0.25]` bounds가 안전하다.

## 다음 후보

1. `EE keyframe pose + RLBench planner`
- PerAct/RVT/RVT-2에 가장 가까운 방향.
- `obs.gripper_pose` 기반 target 생성, quaternion/rotation bin 또는 continuous pose head, planning execution이 필요하다.

2. `relative joint target + JV servo`
- benchmark JV interface를 유지해야 할 때의 후보.
- direct JointPosition 성공률이 먼저 확인되면 같은 target을 JV servo로 바꿔 비교한다.

3. `phase-conditioned action`
- close/insert/scoop처럼 substage가 뚜렷한 task에서 gripper/open-close/keypoint index를 phase token으로 넣는다.
- 현재 0 SR task는 더 오래 학습보다 phase/action representation 문제가 더 커 보인다.
