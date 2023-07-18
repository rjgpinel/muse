import itertools
import math
import numpy as np

from muse.agent.oracle import OracleAgent
from muse.core import constants
from muse.core.utils import (
    quat_to_euler,
    mul_quat,
    euler_to_quat,
    muj_to_std_quat,
    inv_quat,
)
from muse.script import Script
from muse.oracles.utils import get_theta_from_vector


class SweepOracle(OracleAgent):
    def __init__(
        self,
        env,
        np_random,
        gripper_name,
        num_piles,
        piles_size=0.075,
        broom_size=0.03,
        safe_height=0.07,
    ):
        super().__init__(env)
        self._np_random = np_random
        self.scene = env.scene
        self.gripper_name = gripper_name
        self.num_piles = num_piles
        self.script = Script(
            self.scene, self.gripper_name, velocity_profile="trapezoid"
        )
        self.safe_height = safe_height
        self.distance_prepush = piles_size + 0.02
        self.margin_dist = 0.05
        self.dist_to_table_factor = 1.5
        self.piles_size = piles_size
        self.broom_size = broom_size
        self.steps = iter([])
        self.reset()

    def reset(self):
        self.state = "move_to_prepush"
        self.last_gripper_action = constants.GRIP_CLOSE
        self.steps = iter([])

    def move_to_prepush(self, obs):
        max_dist = 0
        max_id = None
        for i in range(self.num_piles):
            pile_dist = obs[f"pile{i}_goal_dist"]
            if pile_dist > max_dist:
                max_dist = pile_dist
                max_id = i

        pile_pos = obs[f"pile{max_id}_pos"]
        pile_safe_pos = pile_pos.copy()
        pile_safe_pos[-1] = self.safe_height
        goal_pos = obs[f"marker0_pos"]

        pushing_vec = goal_pos[:2] - pile_pos[:2]
        norm_pushing_vec = pushing_vec / np.linalg.norm(pushing_vec)
        pushing_dir = get_theta_from_vector(norm_pushing_vec)
        dist_prepushing_vec = -norm_pushing_vec * self.distance_prepush

        push_euler = np.zeros((3))
        push_euler[-1] = pushing_dir
        push_quat = euler_to_quat(push_euler, False)

        orig_gripper_quat = euler_to_quat((0, 180, 180), degrees=True)
        gripper_safe_pos = self.scene.get_ee_pos(self.gripper_name)
        gripper_safe_pos[-1] = self.safe_height
        goal_quat = mul_quat(push_quat, orig_gripper_quat)

        pile_pos[-1] = 0.0

        steps = [
            self.script.tool_move(gripper_safe_pos, quat=orig_gripper_quat),
            self.script.tool_move(
                pile_safe_pos + [dist_prepushing_vec[0], dist_prepushing_vec[1], 0.0],
                quat=goal_quat,
            ),
            self.script.tool_move(
                pile_pos
                + [
                    dist_prepushing_vec[0],
                    dist_prepushing_vec[1],
                    self.broom_size * self.dist_to_table_factor,
                ],
            ),
        ]
        return steps

    def push(self, obs):
        gripper_pos = self.scene.get_ee_pos(self.gripper_name)
        goal_pos = obs[f"marker0_pos"]
        pushing_vec = goal_pos[:2] - gripper_pos[:2]
        norm_pushing_vec = pushing_vec / np.linalg.norm(pushing_vec)
        dist_pushing_vec = -norm_pushing_vec * self.margin_dist
        steps = [
            self.script.tool_move(
                goal_pos
                + [
                    dist_pushing_vec[0],
                    dist_pushing_vec[1],
                    self.broom_size * self.dist_to_table_factor,
                ],
            ),
        ]
        return steps

    def _compute_steps(self, obs):
        if self.state == "move_to_prepush":
            self.steps = itertools.chain(*self.move_to_prepush(obs))
            self.state = "push"
        elif self.state == "push":
            self.steps = itertools.chain(*self.push(obs))
            self.state = "move_to_prepush"
        elif self.state == "end":
            self.steps = iter([])
        else:
            raise ValueError(f"State {self.state} is unknown.")

    def get_action(self, obs):
        action = super().get_action(obs)
        if action is not None:
            angular_velocity = action.pop("angular_velocity")
            theta_velocity = np.array([angular_velocity[-1]])
            action["theta_velocity"] = theta_velocity
        return action
