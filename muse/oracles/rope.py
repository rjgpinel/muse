import itertools

import numpy as np

from muse.core import constants
from muse.core.utils import (
    quat_to_euler,
    mul_quat,
    euler_to_quat,
    muj_to_std_quat,
    inv_quat,
)
from muse.script import Script
from muse.agent.oracle import OracleAgent
from muse.oracles.utils import get_theta_from_vector


class RopeOracle(OracleAgent):
    def __init__(self, env, np_random, gripper_name, rope_num_parts):
        super().__init__(env)
        self._np_random = np_random
        self.scene = env.scene
        self.gripper_name = gripper_name
        self.rope_num_parts = rope_num_parts
        self.height_transport = 0.06
        self.height_release = 0.06
        self.dist_success = env.dist_success
        self.rope_parts_name = env.rope_parts_name
        self.marker0_name = env.marker0_name
        self.marker1_name = env.marker1_name
        self.script = Script(
            self.scene, self.gripper_name, velocity_profile="trapezoid"
        )
        self.steps = iter([])
        self.reset()

    def reset(self):
        self.state = "check"
        self.last_gripper_action = constants.GRIP_OPEN
        self.steps = iter([])

    def pick_part(self, obs, name):
        grasping_pos = obs[f"{name}_pos"]
        grasping_quat = obs[f"{name}_quat"]
        gripper_quat = muj_to_std_quat(self.scene.get_ee_quat(self.gripper_name))
        goal_quat = mul_quat(grasping_quat, gripper_quat)

        steps = [
            self.script.tool_move(
                grasping_pos + [0, 0, self.height_transport], quat=goal_quat
            ),
            self.script.tool_move(grasping_pos + [0, 0, 0]),
            self.script.grip_close(),
            self.script.tool_move(grasping_pos + [0, 0, self.height_transport]),
        ]
        return steps

    def place_part(self, obs, part_name, goal_name):
        placing_pos = obs[f"{goal_name}_pos"]
        grasped_pos = obs[f"{part_name}_pos"]
        orig_gripper_quat = euler_to_quat((0, 180, 90), degrees=True)
        # theta = get_theta_from_vector(placing_pos[:2] - grasped_pos[:2])
        # placing_orn = np.zeros(3)
        # placing_orn[-1] = theta
        # placing_quat = euler_to_quat(placing_orn, False)
        # goal_quat = mul_quat(placing_quat, gripper_quat)

        steps = [
            self.script.tool_move(
                placing_pos + [0, 0, self.height_transport], quat=orig_gripper_quat
            ),
            self.script.tool_move(placing_pos + [0, 0, self.height_release]),
            self.script.grip_open(),
        ]
        return steps

    def _compute_steps(self, obs):
        if self.state == "check":
            end0_pos = obs[f"{self.rope_parts_name[0]}_pos"]
            end1_pos = obs[f"{self.rope_parts_name[-1]}_pos"]
            marker0_pos = obs[f"{self.marker0_name}_pos"]
            marker1_pos = obs[f"{self.marker1_name}_pos"]

            end0_dist = np.linalg.norm(end0_pos[:2] - marker0_pos[:2])
            end0_failure = end0_dist >= self.dist_success

            end1_dist = np.linalg.norm(end1_pos[:2] - marker1_pos[:2])
            end1_failure = end1_dist >= self.dist_success

            if end0_failure or end1_failure:
                if end1_dist > end0_dist:
                    self.state = "pick_end1"
                else:
                    self.state = "pick_end0"
            else:
                self.state = "end"

        if self.state == "pick_end0":
            self.steps = itertools.chain(*self.pick_part(obs, "B0"))
            self.state = "place_end0"
        elif self.state == "place_end0":
            self.steps = itertools.chain(*self.place_part(obs, "B0", "marker0"))
            self.state = "check"
        elif self.state == "pick_end1":
            self.steps = itertools.chain(
                *self.pick_part(obs, f"B{self.rope_num_parts-1}")
            )
            self.state = "place_end1"
        elif self.state == "place_end1":
            self.steps = itertools.chain(
                *self.place_part(obs, f"B{self.rope_num_parts-1}", "marker1")
            )
            self.state = "check"
        elif self.state == "end":
            self.steps = iter([])
        else:
            raise ValueError(f"State {self.state} is unknown.")

    def get_action(self, obs):
        action = super().get_action(obs)
        if action is not None:
            angular_velocity = action.pop("angular_velocity", np.zeros(3))
            theta_velocity = np.array([angular_velocity[-1]])
            action["theta_velocity"] = theta_velocity
        return action
