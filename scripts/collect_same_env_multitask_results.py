import argparse
import json
from pathlib import Path


TASKS = [
    "close_box",
    "insert_onto_square_peg",
    "scoop_with_spatula",
    "slide_block_to_target",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--result-root",
        default="/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite",
    )
    args = parser.parse_args()

    root = Path(args.result_root)
    rows = []
    for task in TASKS:
        result_path = (
            root
            / task
            / f"{task}_{args.run_prefix}_e{args.epochs}_s{args.seed}"
            / "env_000"
            / "result.json"
        )
        if not result_path.exists():
            rows.append(
                {
                    "task": task,
                    "status": "missing",
                    "result_path": str(result_path),
                }
            )
            continue
        data = json.loads(result_path.read_text())
        rows.append(
            {
                "task": task,
                "sr": data.get("sr"),
                "rmse": data.get("offline_seen_env", {}).get("rmse"),
                "result_path": str(result_path),
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
