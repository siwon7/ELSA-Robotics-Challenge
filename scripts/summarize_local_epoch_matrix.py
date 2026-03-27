import json
import math
from pathlib import Path


ROOT = Path("/home/cv25/siwon/ELSA-Robotics-Challenge")
TASKS = [
    "close_box",
    "slide_block_to_target",
    "insert_onto_square_peg",
    "scoop_with_spatula",
]
LOCAL_EPOCHS = [5, 25, 50, 100]
ROUND_NUM = 100


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    rows = []
    for le in LOCAL_EPOCHS:
        for task in TASKS:
            offline_path = (
                ROOT
                / "results"
                / task
                / f"BCPolicy_l-ep_{le}_ts_0.9_fclients_0.05"
                / "results.json"
            )
            sr_path = ROOT / "results" / task / f"sr_le{le}_round_{ROUND_NUM}.json"

            offline_blob = load_json(offline_path)
            sr_blob = load_json(sr_path)

            offline = None
            if offline_blob:
                matches = [item for item in offline_blob if item.get("round") == ROUND_NUM]
                if matches:
                    offline = matches[0]["offline"]

            mse = offline["mean_loss"] if offline else None
            rmse = math.sqrt(mse) if mse is not None else None
            std_mse = offline["std_loss"] if offline else None
            sr = sr_blob["mean_reward"] if sr_blob else None
            std_sr = sr_blob["std_reward"] if sr_blob else None

            rows.append(
                {
                    "task": task,
                    "local_epochs": le,
                    "round": ROUND_NUM,
                    "mse": mse,
                    "rmse": rmse,
                    "std_mse": std_mse,
                    "sr": sr,
                    "std_sr": std_sr,
                }
            )

    out_path = ROOT / "results" / "local_epoch_matrix_summary.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    print(f"saved_to={out_path}")


if __name__ == "__main__":
    main()
