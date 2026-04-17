from bisect import bisect_right

import numpy as np


def _is_stopped(trajectory, index, stopped_buffer, delta=0.1):
    # PerAct-style heuristic: mark a keypoint when the arm is effectively at rest
    # and the gripper state has not changed over a short temporal window.
    if index < 2 or index >= len(trajectory) - 1:
        return False

    obs = trajectory[index]
    next_is_final = index == (len(trajectory) - 2)
    gripper_state_no_change = (
        obs.gripper_open == trajectory[index + 1].gripper_open
        and obs.gripper_open == trajectory[index - 1].gripper_open
        and trajectory[index - 2].gripper_open == trajectory[index - 1].gripper_open
    )
    small_delta = np.allclose(
        np.asarray(obs.joint_velocities, dtype=np.float32),
        0.0,
        atol=delta,
    )
    return (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_final)
        and gripper_state_no_change
    )


def discover_heuristic_keypoints(
    trajectory,
    stopping_delta=0.1,
    stopped_buffer_steps=4,
):
    keypoints = []
    prev_gripper_open = trajectory[0].gripper_open
    stopped_buffer = 0

    for index, obs in enumerate(trajectory):
        stopped = _is_stopped(
            trajectory,
            index,
            stopped_buffer=stopped_buffer,
            delta=stopping_delta,
        )
        stopped_buffer = stopped_buffer_steps if stopped else stopped_buffer - 1
        is_last = index == (len(trajectory) - 1)
        if index != 0 and (obs.gripper_open != prev_gripper_open or is_last or stopped):
            keypoints.append(index)
        prev_gripper_open = obs.gripper_open

    if not keypoints or keypoints[-1] != len(trajectory) - 1:
        keypoints.append(len(trajectory) - 1)

    if len(keypoints) > 1 and (keypoints[-1] - 1) == keypoints[-2]:
        keypoints.pop(-2)

    return keypoints


def find_next_keypoint_index(keypoints, time_step, fallback_last_index):
    keypoint_pos = bisect_right(keypoints, time_step)
    if keypoint_pos >= len(keypoints):
        return fallback_last_index
    return keypoints[keypoint_pos]
