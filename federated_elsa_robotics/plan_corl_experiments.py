"""Plan 4-task CoRL-style FL runs without touching reserved GPU slots."""

from __future__ import annotations

import argparse


TASKS = [
    "slide_block_to_target",
    "close_box",
    "insert_onto_square_peg",
    "scoop_with_spatula",
]


PROFILES = {
    "baseline_corl": {
        "num-server-rounds": 15,
        "local-epochs": 10,
        "fraction-fit": 0.05,
        "train-split": 0.9,
        "use-wandb": "true",
    },
    "baseline_fast": {
        "num-server-rounds": 5,
        "local-epochs": 5,
        "fraction-fit": 0.05,
        "train-split": 0.9,
        "use-wandb": "false",
    },
}


def build_run_command(task: str, slot: int, profile_name: str, run_idx: int) -> str:
    profile = PROFILES[profile_name]
    run_tag = f"slot{slot}_{task}_{profile_name}_run{run_idx}"
    overrides = [
        f"dataset-task='{task}'",
        f"num-server-rounds={profile['num-server-rounds']}",
        f"local-epochs={profile['local-epochs']}",
        f"fraction-fit={profile['fraction-fit']}",
        f"train-split={profile['train-split']}",
        f"use-wandb={profile['use-wandb']}",
        "server-device='cuda:0'",
        "client-device='cuda:0'",
        f"run-tag='{run_tag}'",
    ]
    override_blob = " ".join(overrides)
    return f"CUDA_VISIBLE_DEVICES={slot} flwr run . --run-config \"{override_blob}\""


def main():
    parser = argparse.ArgumentParser(description="Print a safe 4-task FL run plan.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="baseline_corl",
        help="Experiment profile to use for the printed commands.",
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Visible GPU slots to use. Defaults to 2 and 3 so slots 0 and 1 stay untouched.",
    )
    parser.add_argument(
        "--run-idx",
        type=int,
        default=0,
        help="Optional run index to include in the run-tag.",
    )
    args = parser.parse_args()

    print("# Preflight integrity checks")
    for task in TASKS:
        print(
            "python -m federated_elsa_robotics.check_dataset_integrity "
            f"--root ./datasets/training --task {task} --retries 3"
        )

    print("\n# Training commands")
    for idx, task in enumerate(TASKS):
        slot = args.gpu_slots[idx % len(args.gpu_slots)]
        print(build_run_command(task, slot, args.profile, args.run_idx))


if __name__ == "__main__":
    main()
