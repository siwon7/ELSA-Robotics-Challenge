import numpy as np


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    if i < 2:
        return False
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
        i < (len(demo) - 2)
        and (
            obs.gripper_open == demo[i + 1].gripper_open
            and obs.gripper_open == demo[i - 1].gripper_open
            and demo[i - 2].gripper_open == demo[i - 1].gripper_open
        )
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )


def heuristic_keypoint_discovery(demo, stopping_delta=0.1):
    """PerAct-style heuristic keypoint discovery.

    A keypoint is added when the gripper state changes, the arm appears stopped,
    or the episode terminates.
    """
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    return episode_keypoints


def next_keypoint_index(keypoints, time_step, final_index):
    """Return the first keypoint after the current step, else the final step."""
    for keypoint in keypoints:
        if keypoint > time_step:
            return keypoint
    return final_index
