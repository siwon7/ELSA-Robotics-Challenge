import numpy as np


NUM_ARM_JOINTS = 7
EE_POS_DIM = 3
EE_ROT6D_DIM = 6
EE_FEATURE_DIM = EE_POS_DIM + EE_ROT6D_DIM
LOW_DIM_STATE_DIM = NUM_ARM_JOINTS + EE_FEATURE_DIM + 1


def _dh_transform(a, d, alpha, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def panda_forward_kinematics(joint_positions):
    """Return the Panda end-effector transform in the robot base frame."""
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.shape[-1] != NUM_ARM_JOINTS:
        raise ValueError(
            f"Expected {NUM_ARM_JOINTS} joint values, got shape {q.shape}"
        )

    # Franka Panda DH parameters with a fixed flange transform.
    dh_params = [
        (0.0, 0.333, 0.0, q[0]),
        (0.0, 0.0, -np.pi / 2.0, q[1]),
        (0.0, 0.316, np.pi / 2.0, q[2]),
        (0.0825, 0.0, np.pi / 2.0, q[3]),
        (-0.0825, 0.384, -np.pi / 2.0, q[4]),
        (0.0, 0.0, np.pi / 2.0, q[5]),
        (0.088, 0.0, np.pi / 2.0, q[6]),
        (0.0, 0.107, 0.0, 0.0),
    ]

    transform = np.eye(4, dtype=np.float64)
    for a, d, alpha, theta in dh_params:
        transform = transform @ _dh_transform(a, d, alpha, theta)

    return transform.astype(np.float32)


def rotation_matrix_to_rot6d(rotation_matrix):
    """Use the first two rotation matrix columns as a continuous 6D rotation."""
    return rotation_matrix[:, :2].reshape(-1).astype(np.float32)


def build_low_dim_state(joint_positions, gripper_open):
    ee_transform = panda_forward_kinematics(joint_positions)
    ee_position = ee_transform[:3, 3]
    ee_rot6d = rotation_matrix_to_rot6d(ee_transform[:3, :3])
    return np.concatenate(
        (
            np.asarray(joint_positions, dtype=np.float32),
            ee_position,
            ee_rot6d,
            np.array([gripper_open], dtype=np.float32),
        )
    )
