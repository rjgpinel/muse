import numpy as np

from copy import copy
from muse.core import constants
from muse.envs.base import BaseEnv
from muse.envs.utils import random_xy, create_action_space
from muse.oracles.push import PushOracle

from muse.core.modder import DynamicsModder


class PushEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="push_env.xml", **kwargs)
        self.cube_name = "cube0_joint"
        self.goal_name = "marker0"

        self.scene.max_tool_velocity = constants.MAX_TOOL_PUSH_VELOCITY

        workspace_x_center = (
            1 * (self.workspace[1][0] - self.workspace[0][0]) / 8
        ) + self.workspace[0][0]

        self.obj_workspace = self.workspace.copy()
        self.obj_workspace[0, 0] = workspace_x_center
        self.obj_workspace += np.array([[0.2, 0.08, 0.01], [-0.05, -0.08, -0.01]])

        print(f"Obj ws: {self.obj_workspace}")

        self.goal_workspace = self.workspace.copy()
        self.goal_workspace[1, 0] = workspace_x_center
        self.goal_workspace += np.array([[0.0, 0.08, 0.01], [-0.01, -0.08, -0.01]])
        print(f"Goal ws: {self.goal_workspace}")

        self.cube_color = [0.502, 0.769, 0.388]
        self.marker_color = [0.592, 0.188, 0.365]

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
            body_names=["cube0"],
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

    # TODO: clean and remove this func
    def print_params(self):
        # cube_geom_id = self.scene.sim.model.geom_name2id("cube0")
        joint_id = self.scene.sim.model.joint_name2id("cube0_joint")
        print(f"cube frictions: {self.scene.sim.model.dof_frictionloss[joint_id]}")
        print()

    def reset(self):
        super().reset(mocap_target_pos=self._initial_gripper_pos, open_gripper=False)
        self.dynamics_modder.randomize()

        num_cubes = 1
        cube_qpos = random_xy(
            num_cubes, self._np_random, self.obj_workspace, z_pos=0.03
        )
        self.scene.set_joint_qpos(self.cube_name, cube_qpos[0])

        goal_qpos = random_xy(
            num_cubes, self._np_random, self.goal_workspace, z_pos=0.012
        )

        self.scene.sim.data.set_mocap_pos("marker0", goal_qpos[0, :3])
        self.scene.sim.data.set_mocap_quat("marker0", goal_qpos[0, 3:])

        if self.scene.modder.domain_randomization:
            self.scene.modder.rand_hsv("cube0", self.cube_color)
            self.scene.modder.rand_hsv("marker0", self.marker_color)
        else:
            r, g, b = self.cube_color
            cube_rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb("cube0", cube_rgb)

            r, g, b = self.marker_color
            marker_rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb("marker0", marker_rgb)

        self.scene.warmup()
        obs = self.observe()
        return obs

    def observe(self):
        obs = super().observe()
        obs["cube0_pos"] = self.scene.get_site_pos("cube0")
        obs["cube0_quat"] = self.scene.get_body_quat("cube0")
        obs["goal0_pos"] = self.scene.get_site_pos("marker0")
        return obs

    def step(self, action):
        action = copy(action)
        linear_velocity = np.zeros(3)
        linear_velocity[:2] = action.pop("xy_linear_velocity")
        action["linear_velocity"] = linear_velocity
        return super().step(action)

    def is_success(self):
        cube_qpos = self.scene.get_joint_qpos(self.cube_name)
        goal_qpos = self.scene.get_site_pos("marker0")
        success = (
            abs(goal_qpos[0] - cube_qpos[0]) < self.min_x_distance
            and abs(goal_qpos[1] - cube_qpos[1]) < self.min_y_distance
        )
        return success

    def oracle(self):
        return PushOracle(self._np_random)
