# Live SR Eval Commit Handoff

This note is for collaborators who want to pull the reusable live SR evaluation workflow into their own branch.

## Included commit

- `6f53b4f`
  Add reusable live SR evaluation workflow

That commit adds:

- reusable online SR runner
- reusable offline RMSE runner
- local Xvfb / headless live-eval setup
- checkpoint watchdog for automatic SR evaluation
- FKCameraObjectPolicy checkpoint loading support

## Recommended way to take the change

If you only want the SR workflow on top of your current branch:

```bash
git remote add siwon https://github.com/siwon7/ELSA-Robotics-Challenge.git
git fetch siwon main
git cherry-pick 6f53b4f
```

If `siwon` remote already exists:

```bash
git fetch siwon main
git cherry-pick 6f53b4f
```

## If you want to inspect the code first

Open these files:

- `docs/live_sr_eval.md`
- `scripts/run_eval_checkpoint_online.sh`
- `scripts/run_eval_checkpoint_offline.sh`
- `scripts/watchdog_fk_eval.py`
- `scripts/start_watchdog_live_eval_tmux.sh`

## Minimal usage after cherry-pick

Single online SR run:

```bash
scripts/run_eval_checkpoint_online.sh \
  model_checkpoints/close_box/fedavg_FKCameraObjectPolicy_l-ep_25_ts_0.9_fclients_0.05_round_10.pth \
  close_box \
  results/live_eval/close_box_round_10.json \
  eval
```

Watch checkpoints and auto-run SR:

```bash
ELSA_LIVE_EVAL_ROUNDS=5,10,15 \
ELSA_LIVE_EVAL_LOCAL_EPOCHS=25 \
ELSA_LIVE_EVAL_FRACTION_FIT=0.05 \
python scripts/watchdog_fk_eval.py
```

Or start the watchdog in tmux:

```bash
scripts/start_watchdog_live_eval_tmux.sh
```

## Expected outputs

- `results/live_eval/*.json`
- `results/live_eval/watchdog_status.json`
- `logs/live_eval/`

## Notes

- The workflow supports `FKCameraObjectPolicy` checkpoints directly.
- Policy type is inferred from checkpoint names.
- Cached-feature policies are also supported through the same evaluation entrypoint.
