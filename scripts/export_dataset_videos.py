import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from elsa_learning_agent.dataset.compat import load_pickled_data


def load_demo_episodes(root_dir, task, env_id):
    data_path = os.path.join(
        root_dir, task, f"env_{env_id}", "episodes_observations.pkl.gz"
    )
    demos = load_pickled_data(data_path)
    if hasattr(demos, "data"):
        demos = demos.data
    return data_path, demos


def write_mp4_video(frames, output_path, fps):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_avi_video(frames, output_path, fps):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_gif_video(frames, output_path, fps):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(round(1000.0 / float(fps))), 1)
    imageio.mimsave(str(output_path), frames, format="GIF", duration=duration_ms, loop=0)


def write_videos(frames, base_path, fps):
    mp4_path = base_path.with_suffix(".mp4")
    avi_path = base_path.with_suffix(".avi")
    gif_path = base_path.with_suffix(".gif")
    write_mp4_video(frames, mp4_path, fps)
    write_avi_video(frames, avi_path, fps)
    write_gif_video(frames, gif_path, fps)
    return {
        "mp4": str(mp4_path),
        "avi": str(avi_path),
        "gif": str(gif_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-indices", type=int, nargs="*")
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    data_path, demos = load_demo_episodes(args.root_dir, args.task, args.env_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.episode_indices:
        selected_indices = args.episode_indices
    else:
        selected_indices = list(range(min(args.max_episodes, len(demos))))

    summary = {
        "task": args.task,
        "env_id": args.env_id,
        "data_path": data_path,
        "num_episodes_available": len(demos),
        "selected_episode_indices": selected_indices,
        "fps": args.fps,
        "episodes": [],
    }

    for episode_idx in selected_indices:
        demo = demos[episode_idx]
        frames = [obs.front_rgb.copy() for obs in demo]
        video_base = (
            output_dir / "videos" / f"{args.task}_env_{args.env_id}_episode_{episode_idx}"
        )
        video_paths = write_videos(frames, video_base, fps=args.fps)
        episode_summary = {
            "episode_idx": episode_idx,
            "num_steps": len(demo),
            "frame_shape": list(frames[0].shape) if frames else None,
            "video_paths": video_paths,
        }
        summary["episodes"].append(episode_summary)
        print(
            f"episode={episode_idx} steps={len(demo)} "
            f"shape={episode_summary['frame_shape']} gif={video_paths['gif']}"
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
