# Temporal Gripper / Action Contract 리뷰 메모 - 2026-05-07

현재 큐(`ralph_fill4_power_moved_20260507`)는 건드리지 않고, handoff 문서와 archived result를 기준으로 새 관점의 강점과 반례를 정리했다.

## 결론

`temporal gripper event grounding + action/controller contract`는 다음 실험의 좋은 중심 가설이다. 다만 지금 문장처럼 "non-slide 공통 병목은 gripper timing"이라고 단정하면 위험하다. archived 결과에는 `close_box`와 `scoop_with_spatula`의 non-zero 신호가 있고, `insert_onto_square_peg`는 gripper timing 외에 phase/contact geometry 병목일 가능성이 크다.

따라서 논문/제안서 문장은 아래처럼 방어적으로 잡는다.

- 현재 결과는 temporal gripper supervision과 action/execution contract mismatch가 모두 성능 병목일 가능성과 일치한다.
- 두 요인의 상대 기여는 matched ablation으로 분리해야 한다.
- replay ceiling은 expert trajectory executability를 보여주지만 learned closed-loop robustness를 보장하지 않는다.
- insert의 지속적 실패는 gripper timing 외에 phase decomposition 또는 contact-geometry localization이 필요할 수 있음을 시사한다.

## 주요 반례

| 반례 | 의미 | 확인해야 할 것 |
| --- | --- | --- |
| archived `close_box` JV direct가 `0.40-0.55`까지 나온 기록 | non-slide 전체가 항상 0이라는 서술은 선택 편향 가능성 | 최신 코드/동일 eval path에서 재현되는지 matched rerun |
| archived `scoop` JV direct `0.20`, keyframe `0.10` | scoop은 gripper만이 아니라 action family 효과가 있을 수 있음 | grip-weighted BCE와 keyframe/phase baseline 직접 비교 |
| split gripper head가 있어도 close JV direct와 JP-servo가 갈림 | controller contract가 독립 병목일 수 있음 | 같은 checkpoint의 contract swap eval |
| slide diagnostics의 collapse/state-dominance | gripper timing보다 open-loop collapse/replanning 문제가 클 수 있음 | close/insert/scoop에도 same-env diagnostics 확장 |
| insert는 action/vision variant에서 계속 0 | timing만으로는 설명 부족 | oracle gripper vs oracle arm counterfactual 또는 phase/contact supervision |
| mixed-era result 비교 | eval config drift와 split 수정 전후 결과가 섞일 수 있음 | 핵심 baseline과 후보법을 같은 코드 경로로 재실행 |

## 판별 실험

우선순위는 새 학습보다 eval-only 진단이다.

1. Transition-window diagnostic: global gripper metric과 transition-window metric을 분리한다.
2. Hysteresis A/B: 같은 checkpoint에서 threshold만 바꿔 flip count와 SR 변화를 본다.
3. Same-checkpoint contract swap: policy target은 고정하고 execution contract만 바꾼다.
4. Transition-weighted BCE: 위 진단이 지지될 때만 작은 학습 묶음으로 검증한다.
5. Insert/scoop phase/contact 대안: weighted BCE가 안 먹히면 phase/event/interaction point로 넘어간다.

## 코드 반영 방향

이번 코드 변경은 gripper-event 가설을 확정하려는 것이 아니라, 아래 세 가지를 분리 측정하기 위한 장치다.

- transition 주변 gripper BCE를 별도로 키우는 학습 개입
- rollout-only hysteresis로 execution side effect 측정
- transition-window metric으로 "맞는 frame이 어디서 틀리는지" 측정

새 결과는 기존 `action_ablation_20260504`, `overnight_queue`, `recommended_followups_20260504`와 섞지 말고 `tgac_20260507` family로 따로 모은다.
