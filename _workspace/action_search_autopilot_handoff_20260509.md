# Action Search Autopilot Handoff 2026-05-09

Goal: keep searching for a same-env action plus vision representation that lifts all four tasks toward SR >= 0.9 while the existing training managers keep running.

## Active System

- Autopilot tmux: `tmux attach -t action_search_autopilot_20260509`
- Main GPU manager: `tmux attach -t action_search_manager_20260508`
- Per-GPU managers: `tmux attach -t action_search_manager_gpu0_20260508`, `tmux attach -t action_search_manager_gpu1_20260508`, `tmux attach -t action_search_manager_gpu2_20260508`
- Status: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/action_search_manager_20260508/AUTOPILOT_STATUS.md`
- Logs: `/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/action_search_autopilot_20260509/autopilot.log`

The autopilot does not replace the old manager. It periodically appends new candidate rows to the same TSV queues and starts missing manager sessions.

## Current Evidence

- Replay ceiling says joint-position absolute execution is valid: insert and scoop hit SR 1.0 with `jp_abs_hold1`; scoop also has live pack replay at SR 1.0.
- Learned BC with the first VolumeDP jp-abs chunked config is not enough yet: scoop finished at SR 0.0 and slide at SR 0.05.
- Demo retrieval/trajectory search did not hit the 0.9 target; best screened values were unstable and did not confirm.

## Search Policy

The next queue entries are ordered by evidence and risk:

- First: replay-faithful absolute joint-position hold-1 with same-env low dropout, tight VolumeDP bounds, proprio gated-film, EE auxiliary supervision, high gripper transition weighting, and hysteresis.
- Second: current chunk-4/execute-2 absolute joint-position with the same stabilization changes, to isolate whether the old failure was representation or training/eval gripper handling.
- Third: absolute hold-1 with broad bounds and no EE aux, to avoid over-committing to tight spatial assumptions.
- Fourth: fixed-horizon keyframe target with velocity-servo execution, because it gives the policy an easier coarse reaching target while keeping continuous execution.
- Later: relative joint-position servo/direct and joint-velocity direct fallbacks.

## How To Continue

Run once manually:

```bash
/home/cvlab-dgx/anaconda3/envs/elsa_challenge/bin/python \
  scripts/action_search_autopilot_20260509.py \
  --min-outstanding-per-queue 2 \
  --max-add 8
```

Start or keep the loop running:

```bash
bash scripts/start_action_search_autopilot_20260509_tmux.sh
```

If GPU0 or GPU2 are sleeping after queue exhaustion, either wait for the manager sleep interval or restart only those idle sessions after confirming no train process is active on that GPU.
