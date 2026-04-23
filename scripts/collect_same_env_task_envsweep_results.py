#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--result-root",
        default="/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/same_env_suite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.result_root)
    rows = []
    srs = []

    for env_id in args.env_ids:
        run_name = f"{args.task}_{args.run_prefix}_env{env_id}_e{args.epochs}_s{args.seed}"
        result_path = root / args.task / run_name / f"env_{env_id:03d}" / "result.json"
        if not result_path.exists():
            rows.append(
                {
                    "env_id": env_id,
                    "status": "missing",
                    "result_path": str(result_path),
                }
            )
            continue

        data = json.loads(result_path.read_text())
        sr = data.get("sr")
        rmse = data.get("offline_seen_env", {}).get("rmse")
        rows.append(
            {
                "env_id": env_id,
                "sr": sr,
                "rmse": rmse,
                "result_path": str(result_path),
            }
        )
        if sr is not None:
            srs.append(float(sr))

    payload = {
        "task": args.task,
        "run_prefix": args.run_prefix,
        "epochs": args.epochs,
        "seed": args.seed,
        "env_ids": args.env_ids,
        "rows": rows,
        "mean_sr": mean(srs) if srs else None,
        "std_sr": pstdev(srs) if len(srs) > 1 else 0.0 if len(srs) == 1 else None,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
