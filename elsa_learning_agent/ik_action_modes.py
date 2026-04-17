import numpy as np
from pyrep.errors import ConfigurationError, IKError
from rlbench.action_modes.arm_action_modes import (
    ArmActionMode,
    assert_action_shape,
    assert_unit_quaternion,
    calculate_delta_pose,
)
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.scene import Scene


class EndEffectorPoseViaIKSampling(ArmActionMode):
    """IK action mode using sampling instead of the local Jacobian solver."""

    def __init__(
        self,
        absolute_mode: bool = True,
        frame: str = "world",
        collision_checking: bool = False,
        trials: int = 300,
        max_configs: int = 1,
        distance_threshold: float = 0.65,
        max_time_ms: int = 10,
    ):
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._trials = trials
        self._max_configs = max_configs
        self._distance_threshold = distance_threshold
        self._max_time_ms = max_time_ms
        if frame not in ["world", "end effector"]:
            raise ValueError("Expected frame to one of: 'world, 'end effector'")

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != "end effector":
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == "world" else scene.robot.arm.get_tip()

        try:
            joint_positions = scene.robot.arm.solve_ik_via_sampling(
                action[:3],
                quaternion=action[3:],
                ignore_collisions=not self._collision_checking,
                trials=self._trials,
                max_configs=self._max_configs,
                distance_threshold=self._distance_threshold,
                max_time_ms=self._max_time_ms,
                relative_to=relative_to,
            )
            if getattr(joint_positions, "ndim", 1) > 1:
                joint_positions = joint_positions[0]
            joint_positions = np.asarray(joint_positions, dtype=np.float32)
            scene.robot.arm.set_joint_target_positions(joint_positions)
        except (ConfigurationError, IKError) as exc:
            raise InvalidActionError(
                "Could not perform IK via sampling; target pose remained invalid for "
                "the current robot state. Try limiting/bounding your action space."
            ) from exc

        done = False
        prev_values = None
        while not done:
            scene.step()
            cur_positions = scene.robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving

    def action_shape(self, scene: Scene) -> tuple:
        return (7,)
