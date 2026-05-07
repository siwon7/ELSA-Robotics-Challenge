# ELSA Same-Env Action Representation Mission - 2026-05-08

## Goal

Find action representation + execution adapter choices that reach at least `SR >= 0.9`
on all four same-env training tasks:

- `slide_block_to_target`
- `close_box`
- `insert_onto_square_peg`
- `scoop_with_spatula`

Do not stop at the first weak result. Iterate until each task has a defensible
best action contract or the failure mode proves the bottleneck is not action
representation.

## Current Hypothesis

The immediate bottleneck is the action/controller contract, not the vision
backbone. The search must isolate:

- action representation: joint velocity, joint position absolute, joint position relative, keyframe joint position
- controller adapter: direct joint position, direct joint velocity, joint-position-to-joint-velocity servo
- temporal execution: one-step vs chunked/receding horizon
- gripper behavior: direct metadata target, hysteresis, transition timing

## Active Automation

Primary sweep:

```bash
tmux attach -t demo_action_sweep_full_20260508
tmux attach -t demo_action_sweep_followup_20260508
```

Follow-up supervisor sequence:

1. train split nearest-neighbor sweep
2. full index nearest-neighbor sweep
3. train-long nearest-neighbor sweep
4. trajectory-local follower sweep
5. trajectory-replay follower sweep

Fast GPU ceiling training is blocked until these finish:

```bash
tmux attach -t fast_ceiling_search_wait_20260508
```

## Result Paths

Nearest-neighbor / action sweep:

```bash
/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/demo_action_sweep_20260508/_summary.tsv
/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/demo_action_sweep_20260508/<task>/BEST_ACTION.txt
```

Logs:

```bash
/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/logs/demo_action_sweep_20260508/
```

## Important Scripts

```bash
scripts/eval_demo_retrieval_policy_live.py
scripts/eval_demo_trajectory_policy_live.py
scripts/start_demo_action_sweep_20260508_tmux.sh
scripts/start_demo_action_sweep_followup_20260508_tmux.sh
scripts/start_fast_ceiling_search_20260508_tmux.sh
```

## Interpretation Rules

- If `BEST_ACTION.txt` contains a candidate with `SR >= 0.9`, freeze that task's action contract and queue GPU training with that config.
- If only 5-episode screen reaches high SR but 20-episode confirm collapses, treat the candidate as unstable and inspect rollout episode metrics.
- If all nearest-neighbor variants fail but trajectory follower improves, the issue is temporal coherence/gripper timing.
- If trajectory follower also fails, inspect live reset/demo mismatch and expert reproduction before adding more GPU training.
- Do not spend GPU on vision changes until action contract has a same-env ceiling signal.

## Next Actions For Another Agent

1. Read `_summary.tsv` and all `BEST_ACTION.txt` files.
2. If any task is below `0.9`, inspect the highest-SR result JSON for that task.
3. Add new action candidates only if they test a distinct controller contract.
4. Prefer CPU/live probes first; use GPU training only after an action contract has a clear ceiling.
5. Keep fast ceiling training blocked while action search is still active.
