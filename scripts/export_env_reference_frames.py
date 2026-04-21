#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from elsa_learning_agent.dataset.compat import load_pickled_data


DEFAULT_TASKS = [
    "slide_block_to_target",
    "close_box",
    "insert_onto_square_peg",
    "scoop_with_spatula",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one front_rgb frame per env and build contact sheets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/raid0/siwon/data/ELSA-Robotics-Challenge/datasets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/raid0/siwon/ELSA-Robotics-Challenge-artifacts/results/env_reference_frames"
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
    )
    parser.add_argument(
        "--training-envs",
        nargs="*",
        type=int,
        default=[0],
    )
    parser.add_argument(
        "--eval-envs",
        nargs="*",
        type=int,
        default=list(range(400, 410)),
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=0,
    )
    return parser.parse_args()


def render_label(image: Image.Image, label: str) -> Image.Image:
    framed = image.copy()
    draw = ImageDraw.Draw(framed)
    draw.rectangle((0, 0, framed.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255))
    return framed


def export_single_frame(
    dataset_root: Path,
    output_root: Path,
    split: str,
    task: str,
    env_id: int,
    episode_index: int,
    step_index: int,
) -> Path:
    data_path = dataset_root / split / task / f"env_{env_id}" / "episodes_observations.pkl.gz"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    demos = load_pickled_data(data_path)
    obs = demos[episode_index][step_index]
    image = Image.fromarray(obs.front_rgb)

    task_dir = output_root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = task_dir / f"{split}_env_{env_id:03d}_front_rgb.png"
    image.save(out_path)
    return out_path


def build_contact_sheet(task: str, image_paths: Iterable[Path], output_root: Path) -> Path | None:
    image_paths = list(image_paths)
    if not image_paths:
        return None

    images = [render_label(Image.open(path).convert("RGB"), path.stem) for path in image_paths]
    width, height = images[0].size
    cols = 4
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), color=(30, 30, 30))

    for idx, img in enumerate(images):
        x = (idx % cols) * width
        y = (idx // cols) * height
        sheet.paste(img, (x, y))

    out_path = output_root / task / "contact_sheet.png"
    sheet.save(out_path)
    return out_path


def main() -> None:
    args = parse_args()

    summary_lines = []
    for task in args.tasks:
        exported = []
        for env_id in args.training_envs:
            exported.append(
                export_single_frame(
                    dataset_root=args.dataset_root,
                    output_root=args.output_root,
                    split="training",
                    task=task,
                    env_id=env_id,
                    episode_index=args.episode_index,
                    step_index=args.step_index,
                )
            )
        for env_id in args.eval_envs:
            exported.append(
                export_single_frame(
                    dataset_root=args.dataset_root,
                    output_root=args.output_root,
                    split="eval",
                    task=task,
                    env_id=env_id,
                    episode_index=args.episode_index,
                    step_index=args.step_index,
                )
            )
        sheet_path = build_contact_sheet(task, exported, args.output_root)
        summary_lines.append(f"{task}: {len(exported)} images")
        if sheet_path is not None:
            summary_lines.append(f"  contact_sheet: {sheet_path}")

    summary_path = args.output_root / "README.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
