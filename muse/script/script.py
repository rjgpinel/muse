import numpy as np
import math
from muse.core import constants
from muse.core.utils import (
    quat_to_euler,
    muj_to_std_quat,
    euler_to_quat,
    inv_quat,
    mul_quat,
    get_axis_from_quat,
    get_angle_from_quat,
)
from muse.script.scurve import create_scurve


class Script:
    def __init__(self, scene, tool_name, velocity_profile="scurve"):
        self.dt = scene.controller_dt
        self.max_v, self.max_w = scene.max_tool_velocity
        self.ee_pos = scene.get_ee_pos
        self.ee_quat = scene.get_ee_quat
        self.gripper_action_length = constants.GRIP_ACTION_LENGTH
        self.tool_name = tool_name
        self.velocity_profile = velocity_profile

    def tool_move(self, pos=None, quat=None, t_acc=1.0):
        current_pos = self.ee_pos(self.tool_name)
        current_quat = muj_to_std_quat(self.ee_quat(self.tool_name))
        rel_dist, axis, angle = np.zeros(3), np.zeros(3), 0.0
        if pos is not None:
            rel_dist = pos - current_pos
        if quat is not None:
            quat_dist = mul_quat(quat, inv_quat(current_quat))
            angle = get_angle_from_quat(quat_dist)
            axis = get_axis_from_quat(quat_dist)
            if self.velocity_profile == "scurve":
                raise ValueError("Change of orientation not handled")

        vels = []
        if self.velocity_profile == "trapezoid":
            vel_prof = trapezoidal_velocity_profile(
                [rel_dist, angle * axis], [self.max_v, self.max_w], self.dt, t_acc
            )
            for v, w in vel_prof:
                yield dict(linear_velocity=v, angular_velocity=w)
        elif self.velocity_profile == "scurve":
            vel_prof = scurve_velocity_profile(rel_dist, self.max_v, self.dt)
            for v in vel_prof:
                yield dict(linear_velocity=v)

    def grip_close(self):
        vels = []
        for _ in np.arange(0, self.gripper_action_length, self.dt):
            yield dict(grip_open=constants.GRIP_CLOSE)

    def grip_open(self):
        vels = []
        for _ in np.arange(0, self.gripper_action_length, self.dt):
            yield dict(grip_open=constants.GRIP_OPEN)

    def pause(self):
        PAUSE_LENGTH = 10
        for _ in np.arange(0, 1, self.dt):
            yield dict(linear_velocity=np.zeros(3))


def trapezoidal_velocity_profile(rel_dist, max_velocities, dt, t_acc=1.0):
    t_max = [np.linalg.norm(d / v) for d, v in zip(rel_dist, max_velocities)]

    t_dec = np.max(t_max)
    t_acc = np.min([t_acc, t_dec])
    t_end = t_dec + t_acc
    v_coeff = [d / t_dec for d in rel_dist]

    vels = []
    for t in np.arange(0.0, t_end, dt) + dt:
        k = 1.0
        if t > t_end:
            k = 0.0
        elif t <= t_acc:
            k = t / t_acc
        elif t >= t_dec:
            k = 1 - (t - t_dec) / t_acc
        vels.append([k * v for v in v_coeff])
    return vels


def scurve_velocity_profile(rel_dist, max_velocity, dt):
    # assert np.linalg.norm(rel_dist) > 1e-3
    d = np.linalg.norm(rel_dist)
    p0 = np.zeros(1)
    p1 = d * np.ones(1)
    v_max = max_velocity
    a_max = 4 * max_velocity
    j_max = 4 * max_velocity

    scurve_fn, t_final = create_scurve(p0, p1, v_max, a_max, j_max)
    timesteps = int(np.ceil(max(t_final) / dt))
    time = np.linspace(0, max(t_final), timesteps)
    traj = np.stack([scurve_fn(t) for t in time])
    vel = traj[:, 1]
    vel = vel[:, None] * rel_dist[None, :] / d

    return vel
