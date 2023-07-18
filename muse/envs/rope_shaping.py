import numpy as np

from copy import copy
from muse.core.utils import (
    muj_to_std_quat,
    euler_to_quat,
    quat_to_euler,
    std_to_muj_quat,
)
from muse.envs.base import BaseEnv
from muse.envs.utils import random_xy, create_action_space
from muse.oracles.rope import RopeOracle
from muse.script import Script

from muse.core.modder import DynamicsModder


class RopeShapingEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="rope_shaping_env.xml", **kwargs)
        self.rope_name = "rope0_joint"
        self.main_part = 4
        self.rope_num_parts = 8
        self.rope_parts_name = [f"B{i}" for i in range(self.rope_num_parts)]
        self.marker0_name = "marker0"
        self.marker0_color = [0.365, 0.737, 0.384]
        self.marker1_name = "marker1"
        self.marker1_color = [0.592, 0.188, 0.365]

        self.rope_radius = self.scene.get_geom_size(f"G{self.main_part}")[1]

        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open", "theta_velocity"
        )

        self.rope_workspace = self.workspace.copy()
        self.rope_workspace += np.array([[0.025, 0.08, 0.01], [-0.025, -0.08, -0.01]])
        self.dist_success = 0.04

        self.rope_color = np.array([166, 159, 126]) / 255

    def reset(self):
        gripper_quat = euler_to_quat((0, 180, 90), degrees=True)
        gripper_quat = std_to_muj_quat(gripper_quat)
        super().reset(mocap_target_quat=gripper_quat)

        num_ropes = 1
        valid = False

        while not valid:
            rope_qpos = random_xy(
                num_ropes, self._np_random, self.rope_workspace, z_pos=0.03
            )
            self.scene.set_joint_qpos(self.rope_name, rope_qpos[0])
            for i in range(self.rope_num_parts):
                if i == self.main_part:
                    continue
                part_rot = self._np_random.uniform(-np.pi / 4, np.pi / 4)
                self.scene.sim.data.set_joint_qpos(f"J1_{i}", part_rot)
            end0_pos = self.scene.get_body_pos(self.rope_parts_name[0])
            end1_pos = self.scene.get_body_pos(self.rope_parts_name[-1])

            marker0_pos = self.scene.get_site_pos(self.marker0_name)
            marker1_pos = self.scene.get_site_pos(self.marker1_name)

            # print(f"Marker 0 pos: {marker0_pos}")
            # print(f"Marker 1 pos: {marker1_pos}")

            end0_dist = np.linalg.norm(end0_pos[:2] - marker0_pos[:2])
            end1_dist = np.linalg.norm(end1_pos[:2] - marker1_pos[:2])

            # We need to warmup until the rope touches the table
            self.scene.warmup()
            valid = (
                (
                    (self.obj_workspace[0] < end0_pos)
                    & (end0_pos < self.obj_workspace[1])
                ).all()
                and (
                    (self.obj_workspace[0] < end1_pos)
                    & (end1_pos < self.obj_workspace[1])
                ).all()
                and (
                    (self.obj_workspace[0] < end0_pos)
                    & (end0_pos < self.obj_workspace[1])
                ).all()
                and (
                    (self.obj_workspace[0] < end1_pos)
                    & (end1_pos < self.obj_workspace[1])
                ).all()
                and end0_dist > self.dist_success + 0.02
                and end1_dist > self.dist_success + 0.02
                and not self.is_success()
            )

        if self.scene.modder.domain_randomization:
            self.scene.modder.rand_hsv(f"G{self.main_part}", self.rope_color)
            self.scene.modder.rand_hsv("marker0", self.marker0_color)
            self.scene.modder.rand_hsv("marker1", self.marker1_color)
        else:
            r, g, b = self.rope_color
            rope_rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb(f"G{self.main_part}", rope_rgb)
            r, g, b = self.marker0_color
            marker0_rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb("marker0", marker0_rgb)
            r, g, b = self.marker1_color
            marker1_rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb("marker1", marker1_rgb)

        # # markers positions
        # valid = False
        # while not valid:
        #     markers_qpos = random_xy(
        #         num_markers, self._np_random, self.obj_workspace, z_pos=0.0001
        #     )
        #     valid = self.rope_num_parts * 0.04 > np.linalg.norm(
        #         markers_qpos[0, :3] - markers_qpos[1, :3]
        #     )

        # self.scene.sim.data.set_mocap_pos(self.marker0_name, markers_qpos[0, :3])
        # self.scene.sim.data.set_mocap_quat(self.marker0_name, markers_qpos[0, 3:])

        # self.scene.sim.data.set_mocap_pos(self.marker1_name, markers_qpos[1, :3])
        # self.scene.sim.data.set_mocap_quat(self.marker1_name, markers_qpos[1, 3:])

        self.scene.warmup()
        obs = self.observe()

        return obs

    def observe(self):
        obs = super().observe()
        for i, part_name in enumerate(self.rope_parts_name):
            obs[f"{part_name}_pos"] = self.scene.get_body_pos(part_name)
            part_euler = quat_to_euler(self.scene.get_site_quat(f"S{i}"), True)
            if part_euler[-1] < 0:
                part_euler[-1] += 360

            obs[f"{part_name}_quat"] = euler_to_quat(part_euler, True)
        obs[f"{self.marker0_name}_pos"] = self.scene.get_site_pos(self.marker0_name)
        obs[f"{self.marker1_name}_pos"] = self.scene.get_site_pos(self.marker1_name)
        return obs

    def is_success(self):
        end0_pos = self.scene.get_body_pos(self.rope_parts_name[0])
        end1_pos = self.scene.get_body_pos(self.rope_parts_name[-1])
        marker0_pos = self.scene.get_site_pos(self.marker0_name)
        marker1_pos = self.scene.get_site_pos(self.marker1_name)

        end0_dist = np.linalg.norm(end0_pos[:2] - marker0_pos[:2])
        end1_dist = np.linalg.norm(end1_pos[:2] - marker1_pos[:2])

        success = (
            end0_dist < self.dist_success
            and end1_dist < self.dist_success
            and end0_pos[-1] < self.rope_radius + 0.01
            and end1_pos[-1] < self.rope_radius + 0.01
        )
        return success

    def oracle(self):
        return RopeOracle(self, self._np_random, self.gripper_name, self.rope_num_parts)

    def step(self, action):
        new_action = action.copy()
        new_action["angular_velocity"] = np.zeros(3)
        new_action["angular_velocity"][-1] = new_action.pop("theta_velocity", 0.0)
        return super().step(new_action)
