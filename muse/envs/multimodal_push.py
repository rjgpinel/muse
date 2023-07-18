import numpy as np

from copy import copy
from muse.core import constants
from muse.envs.base import BaseEnv
from muse.envs.utils import random_xy, create_action_space
from muse.oracles.multimodal_push import MultimodalPushOracle

from muse.core.modder import DynamicsModder


class MultimodalPushEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="multimodal_push_env.xml", **kwargs)

        self.num_cubes = 2
        self.cubes_name = [f"cube{i}_joint" for i in range(self.num_cubes)]
        self.goals_name = ["marker0", "marker1"]

        self.scene.max_tool_velocity = constants.MAX_TOOL_PUSH_VELOCITY

        workspace_x_center = (
            1 * (self.workspace[1][0] - self.workspace[0][0]) / 8
        ) + self.workspace[0][0]

        self.obj_workspace = self.workspace.copy()
        self.obj_workspace[0, 0] = workspace_x_center
        self.obj_workspace += np.array([[0.2, 0.08, 0.01], [-0.1, -0.08, -0.01]])

        self.goal_workspace = self.workspace.copy()
        self.goal_workspace[1, 0] = workspace_x_center
        self.goal_workspace += np.array([[0.0, 0.08, 0.01], [-0.01, -0.08, -0.01]])

        self.cubes_color = [(0.502, 0.769, 0.388), (0.502, 0.769, 0.388)]
        self.markers_color = [(0.592, 0.188, 0.365), (0.592, 0.188, 0.365)]

        self.default_gripper_height = 0.035
        self.min_x_distance = 0.02
        self.min_y_distance = 0.02

        self._initial_gripper_pos = np.array([self.workspace[1][0], 0.0, 0.0])
        self._initial_gripper_pos[-1] = self.default_gripper_height

        self.action_space = create_action_space(self.scene, "xy_linear_velocity")

        self.dynamics_modder = DynamicsModder(
            self.scene.sim,
            random_state=self._np_random,
            # Opt parameters
            randomize_density=True,
            randomize_viscosity=True,
            density_perturbation_ratio=0.3,
            viscosity_perturbation_ratio=0.3,
            # Body parameters
            body_names=["cube0", "cube1"],
            randomize_position=False,
            randomize_quaternion=False,
            randomize_inertia=True,
            randomize_mass=True,
            # Geom parameters
            randomize_friction=False,
            randomize_solref=False,
            randomize_solimp=False,
            # Joint parameters
            joint_names=["cube0_joint"],
            randomize_stiffness=True,
            randomize_frictionloss=True,
            randomize_damping=True,
            randomize_armature=True,
            frictionloss_perturbation_size=0.2,
        )

    def reset(self):
        super().reset(mocap_target_pos=self._initial_gripper_pos, open_gripper=False)
        # self.dynamics_modder.randomize()

        valid = False

        while not valid:
            cubes_qpos = random_xy(
                self.num_cubes,
                self._np_random,
                self.obj_workspace,
                z_pos=0.03,
            )
            valid = True
            for i in range(1, self.num_cubes):
                valid = np.abs(cubes_qpos[i][1] - cubes_qpos[i - 1][1]) > 0.06
                if not valid:
                    break

        for i in range(self.num_cubes):
            self.scene.set_joint_qpos(self.cubes_name[i], cubes_qpos[i])

        # goal_qpos = random_xy(1, self._np_random, self.goal_workspace, z_pos=0.012)

        # self.scene.sim.data.set_mocap_pos("markers", goal_qpos[0, :3])
        # self.scene.sim.data.set_mocap_quat("markers", goal_qpos[0, 3:])

        if self.scene.modder.domain_randomization:
            for i in range(self.num_cubes):
                self.scene.modder.rand_hsv(f"cube{i}", self.cubes_color[i])
            for i in range(len(self.goals_name)):
                self.scene.modder.rand_hsv(f"marker{i}", self.markers_color[i])
        else:
            for i in range(self.num_cubes):
                r, g, b = self.cubes_color[i]
                cube_rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
                self.scene.modder.texture_modder.set_rgb(f"cube{i}", cube_rgb)

            for i in range(len(self.goals_name)):
                r, g, b = self.markers_color[i]
                marker_rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
                self.scene.modder.texture_modder.set_rgb(self.goals_name[i], marker_rgb)

        self.scene.warmup()
        obs = self.observe()
        return obs

    def observe(self):
        obs = super().observe()
        for i in range(self.num_cubes):
            obs[f"cube{i}_pos"] = self.scene.get_site_pos(f"cube{i}")
            obs[f"cube{i}_quat"] = self.scene.get_body_quat(f"cube{i}")
        obs["goal0_pos"] = self.scene.get_site_pos("marker0")
        obs["goal1_pos"] = self.scene.get_site_pos("marker1")
        return obs

    def step(self, action):
        action = copy(action)
        linear_velocity = np.zeros(3)
        linear_velocity[:2] = action.pop("xy_linear_velocity")
        action["linear_velocity"] = linear_velocity
        return super().step(action)

    def is_success(self):
        obj_success = [False] * self.num_cubes
        for i in range(self.num_cubes):
            cube_qpos = self.scene.get_joint_qpos(self.cubes_name[i])
            for goal_name in self.goals_name:
                goal_qpos = self.scene.get_site_pos(goal_name)
                success = (
                    abs(goal_qpos[0] - cube_qpos[0]) < self.min_x_distance
                    and abs(goal_qpos[1] - cube_qpos[1]) < self.min_y_distance
                )
                if success:
                    obj_success[i] = True
                    break
        return np.all(obj_success)

    def oracle(self):
        return MultimodalPushOracle(
            self._np_random, self.min_x_distance, self.min_y_distance
        )
