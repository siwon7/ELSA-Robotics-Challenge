# CoRL Experiment Layout

This repository now keeps the research workflow separate from the default Flower entrypoint.

## Agent split

- `research_director`: owns claims, priorities, and synthesis
- `repro_guardian`: owns dataset integrity, rerun hygiene, and clean-run certification
- `implementer`: the only write agent
- `result_analyst`: owns metric summaries and checkpoint comparisons
- `counterexample_scout`: owns reviewer-style criticism

## Task slots

The codebase exposes four tasks, so the current plan treats them as four slots:

- slot `0`: `close_box` (`protected`)
- slot `1`: `insert_onto_square_peg` (`protected`)
- slot `2`: `scoop_with_spatula` (`queued`)
- slot `3`: `slide_block_to_target` (`active`)

Slots `0` and `1` are intentionally left untouched.

## Current execution order

1. `slot3_slide_baseline_repro`
2. `slot3_slide_structured_retry`
3. `slot2_scoop_baseline_pilot`

## Useful commands

List the current plan:

```bash
python -m federated_elsa_robotics.experiment_plan --list
```

Show one experiment:

```bash
python -m federated_elsa_robotics.experiment_plan --experiment slot3_slide_baseline_repro
```

Run the dataset preflight for the active slide task:

```bash
python -m federated_elsa_robotics.repro_guard --slot 3 --split training --retries 3
```

Validate a config/model/dataloader/action combination on CPU before a real run:

```bash
python scripts/validate_experiment_config.py \
  --config experiments/slide_block_to_target_chunk3_dinov3_fedprox_main.yaml \
  --task slide_block_to_target \
  --env-id 0 \
  --split train \
  --normalize
```

Reference docs:
- [FL roadmap](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/fl_experiment_roadmap_kr.md)
- [FL method plan](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/federated_method_plan_kr.md)
- [Action presets](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/action_pipeline_presets_kr.md)
- [Model catalog](/home/cvlab-dgx/siwon/ELSA-Robotics-Challenge/docs/model_catalog_kr.md)
