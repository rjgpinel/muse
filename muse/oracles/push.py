import numpy as np

from muse.core.constants import GRIP_CLOSE, MAX_TOOL_PUSH_VELOCITY, CONTROLLER_DT
from muse.core.utils import muj_to_std_quat, quat_to_euler
from muse.script import Script
from muse.oracles.utils import rotate, theta_to_rotation2d, get_theta_from_vector


class PushOracle:
    def __init__(self, np_random):
        self.theta_threshold_to_orient = 0.025
        self.theta_threshold_flat_enough = 0.0025
        self.theta_rotation = 0.03
        self.orient_circle_diameter = 0.0225
        # self.action_noise_std = 0.001
        self.action_noise_std = 0
        self.state = None
        self.origin = None
        self.first_preblock = None
        self._np_random = np_random
        self._obj_name = "cube0"
        self._goal_name = "goal0"
        self.first_preblock = None

        self.reset()

    def reset(self):
        self.state = "move_to_preblock"

    def get_action(self, obs):
        if self.origin is None:
            self.origin = obs["gripper_pos"][:2]

        if self.state == "move_to_preblock":
            relative_xy = self._move_to_preblock(obs)

        elif self.state == "return_to_first_preblock":
            relative_xy = self._return_to_first_preblock(obs)

        elif self.state == "return_to_origin":
            relative_xy = self._return_to_origin(obs)

        elif self.state == "move_to_block":
            relative_xy = self._move_to_block(obs)

        elif self.state == "push_block":
            relative_xy = self._push_block(obs)

        elif self.state == "orient_block_left":
            relative_xy = self._orient_block(obs, direction="left")

        elif self.state == "orient_block_right":
            relative_xy = self._orient_block(obs, direction="right")

        if self.action_noise_std != 0.0:
            relative_xy += _np_random.randn(2) * self.action_noise_std

        xy_linear_velocity = np.zeros(2)

        max_step_distance = MAX_TOOL_PUSH_VELOCITY[0] * CONTROLLER_DT
        length = np.linalg.norm(relative_xy)

        if length > max_step_distance:
            direction_xy = relative_xy / length
            relative_xy = direction_xy * max_step_distance

        xy_linear_velocity = relative_xy / CONTROLLER_DT

        action = dict(xy_linear_velocity=xy_linear_velocity, grip_open=GRIP_CLOSE)
        return action

    def _move_to_preblock(self, obs):
        block_xy = obs[f"{self._obj_name}_pos"][:2]
        goal_xy = obs[f"{self._goal_name}_pos"][:2]
        gripper_xy = obs["gripper_pos"][:2]

        block_to_goal_xy = goal_xy - block_xy
        block_to_goal_xy_dir = block_to_goal_xy / np.linalg.norm(block_to_goal_xy)

        # Go 5 cm away from the block, on the line between the block and target.
        preblock_xy = block_xy - block_to_goal_xy_dir * 0.05
        relative_xy = preblock_xy - gripper_xy
        diff = np.linalg.norm(relative_xy)

        if diff < 0.001:
            self.state = "move_to_block"
            if self.first_preblock is None:
                self.first_preblock = np.copy(preblock_xy)
        return relative_xy

    def _return_to_first_preblock(self, obs):
        gripper_xy = obs["gripper_pos"][:2]

        if self.first_preblock is None:
            self.first_preblock = self.origin
        # Return to the first preblock.
        relative_to_origin = self.first_preblock - gripper_xy
        diff = np.linalg.norm(relative_to_origin)
        if diff < 0.001:
            self.state = "return_to_origin"
        return relative_to_origin

    def _return_to_origin(self, obs):
        # Go 5 cm away from the block, on the line between the block and target.
        gripper_xy = obs["gripper_pos"][:2]
        relative_to_origin = self.origin - gripper_xy
        diff = np.linalg.norm(relative_to_origin)
        if diff < 0.001:
            self.state = "move_to_preblock"
        return relative_to_origin

    def _move_to_block(self, obs):
        block_xy = obs[f"{self._obj_name}_pos"][:2]
        goal_xy = obs[f"{self._goal_name}_pos"][:2]
        gripper_xy = obs["gripper_pos"][:2]

        block_to_goal_xy = goal_xy - block_xy
        block_to_goal_xy_dir = block_to_goal_xy / np.linalg.norm(block_to_goal_xy)
        next_to_block_xy = block_xy - block_to_goal_xy_dir * 0.03
        relative_xy = next_to_block_xy - gripper_xy
        diff = np.linalg.norm(relative_xy)
        if diff < 0.001:
            self.state = "push_block"

        block_theta = quat_to_euler(
            muj_to_std_quat(obs[f"{self._obj_name}_quat"]), degrees=False
        )[-1]
        theta_to_goal = get_theta_from_vector(block_to_goal_xy_dir)
        theta_error = theta_to_goal - block_theta
        # Block has 4-way symmetry.
        while theta_error > np.pi / 4:
            theta_error -= np.pi / 2.0
        while theta_error < -np.pi / 4:
            theta_error += np.pi / 2.0

        # Re-orient the object if needed
        if theta_error > self.theta_threshold_to_orient:
            self.state = "orient_block_left"
        elif theta_error < -self.theta_threshold_to_orient:
            self.state = "orient_block_right"

        return relative_xy

    def _push_block(self, obs):
        block_xy = obs[f"{self._obj_name}_pos"][:2]
        goal_xy = obs[f"{self._goal_name}_pos"][:2]
        gripper_xy = obs["gripper_pos"][:2]

        block_to_goal_xy = goal_xy - block_xy
        block_to_goal_xy_dir = block_to_goal_xy / np.linalg.norm(block_to_goal_xy)

        touching_block_xy = block_xy - block_to_goal_xy_dir * 0.01
        relative_xy = touching_block_xy

        block_theta = quat_to_euler(
            muj_to_std_quat(obs[f"{self._obj_name}_quat"]), degrees=False
        )[-1]
        theta_to_goal = get_theta_from_vector(block_to_goal_xy_dir)
        theta_error = theta_to_goal - block_theta
        # Block has 4-way symmetry.
        while theta_error > np.pi / 4:
            theta_error -= np.pi / 2.0
        while theta_error < -np.pi / 4:
            theta_error += np.pi / 2.0

        # If need to reorient, go back to move_to_pre_block, move_to_block first.
        if theta_error > self.theta_threshold_to_orient:
            self.state = "move_to_preblock"
        elif theta_error < -self.theta_threshold_to_orient:
            self.state = "move_to_preblock"
        return relative_xy

    def _orient_block(self, obs, direction="right"):
        block_xy = obs[f"{self._obj_name}_pos"][:2]
        goal_xy = obs[f"{self._goal_name}_pos"][:2]
        gripper_xy = obs["gripper_pos"][:2]

        block_to_gripper_xy = gripper_xy - block_xy
        block_to_gripper_xy_dir = block_to_gripper_xy / np.linalg.norm(
            block_to_gripper_xy
        )

        if direction == "right":
            theta_rotation = -self.theta_rotation
        elif direction == "left":
            theta_rotation = self.theta_rotation
        else:
            raise ValueError(
                f"Direction {direction} is not valid. Only 'right' or 'left' is allowed."
            )

        block_to_gripper_xy_dir = rotate(theta_rotation, block_to_gripper_xy_dir)
        block_to_gripper_xy = block_to_gripper_xy_dir * self.orient_circle_diameter
        push_left_spot_xy = block_xy + block_to_gripper_xy
        relative_xy = push_left_spot_xy - gripper_xy

        block_to_goal_xy = goal_xy - block_xy
        block_to_goal_xy_dir = block_to_goal_xy / np.linalg.norm(block_to_goal_xy)
        block_theta = quat_to_euler(
            muj_to_std_quat(obs[f"{self._obj_name}_quat"]), degrees=False
        )[-1]
        theta_to_goal = get_theta_from_vector(block_to_goal_xy_dir)
        theta_error = theta_to_goal - block_theta
        # Block has 4-way symmetry.
        while theta_error > np.pi / 4:
            theta_error -= np.pi / 2.0
        while theta_error < -np.pi / 4:
            theta_error += np.pi / 2.0

        if direction == "left" and theta_error < self.theta_threshold_flat_enough:
            self.state = "move_to_preblock"
        if direction == "right" and theta_error > -self.theta_threshold_flat_enough:
            self.state = "move_to_preblock"

        return relative_xy
