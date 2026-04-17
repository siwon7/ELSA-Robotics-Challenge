# CoRL Experiment Setup

This repository currently exposes four RLBench tasks:

- `slide_block_to_target`
- `close_box`
- `insert_onto_square_peg`
- `scoop_with_spatula`

For the current paper workflow, use the following role split:

- `research_director`: decide claim, stop/go, and next experiment
- `repro_guardian`: dataset integrity, failed env tracking, rerun hygiene
- `implementer`: code/config changes only
- `result_analyst`: summarize offline and online metrics
- `counterexample_scout`: reviewer-style criticism and missing baseline checks

## Safe Run Policy

- Keep GPU slots `0` and `1` reserved.
- Launch new FL runs on slots `2` and `3` with `CUDA_VISIBLE_DEVICES=<slot>`.
- Keep the default `flwr run .` path intact; use run-time overrides and `run-tag` for new experiments.

## Recommended Sequence

1. Run integrity preflight for each task.
2. Run the 4-task baseline plan on slots `2` and `3`.
3. Evaluate checkpoints with the same run-tagged configuration.
4. Only after a clean baseline, introduce structured changes.

## Commands

Print the current 4-task training plan:

```bash
python -m federated_elsa_robotics.plan_corl_experiments --profile baseline_corl --gpu-slots 2 3
```

Run integrity checks before training:

```bash
python -m federated_elsa_robotics.check_dataset_integrity --root ./datasets/training --retries 3
```
