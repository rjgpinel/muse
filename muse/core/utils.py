import numpy as np
import mujoco_py
from pathlib import Path
from math import atan2
from scipy.spatial.transform import Rotation


def assets_dir():
    return Path(__file__).parent.parent / "assets"


def xmat_to_std_quat(xmat):
    rotation = Rotation.from_matrix(xmat)
    return rotation.as_quat()


def std_to_muj_quat(quat):
    muj_quat = quat.copy()
    muj_quat[0] = quat[3]
    muj_quat[1:] = quat[:3]
    return muj_quat


def muj_to_std_quat(muj_quat):
    quat = muj_quat.copy()
    quat[3] = muj_quat[0]
    quat[:3] = muj_quat[1:]
    return quat


def euler_to_quat(euler, degrees):
    rotation = Rotation.from_euler("xyz", euler, degrees=degrees)
    return rotation.as_quat()


def quat_to_euler(quat, degrees):
    rotation = Rotation.from_quat(quat)
    return rotation.as_euler("xyz", degrees=degrees)


def get_axis_from_quat(quat, undefined=np.zeros(3)):
    tolerance = 1e-17
    vector = quat[0:3]
    norm = np.linalg.norm(vector)
    if norm < tolerance:
        # infinite set of possible axes
        return undefined
    else:
        return vector / norm


def get_angle_from_quat(quat):
    vector = quat[0:3]
    scalar = quat[-1]
    norm = np.linalg.norm(vector)
    return wrap_angle(2.0 * atan2(norm, scalar))


def wrap_angle(theta):
    result = ((theta + np.pi) % (2 * np.pi)) - np.pi
    if result == -np.pi:
        result = np.pi
    return result


def mul_quat(quat0, quat1):
    rot0 = Rotation.from_quat(quat0)
    rot1 = Rotation.from_quat(quat1)
    rot = rot0 * rot1
    return rot.as_quat()


def inv_quat(quat):
    rotation = Rotation.from_quat(quat)
    inv_rotation = rotation.inv()
    return inv_rotation.as_quat()


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith("robot")]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7,))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7,))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation."""
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (
        sim.model.eq_type is None
        or sim.model.eq_obj1id is None
        or sim.model.eq_obj2id is None
    ):
        return
    for eq_type, obj1_id, obj2_id in zip(
        sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
    ):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
