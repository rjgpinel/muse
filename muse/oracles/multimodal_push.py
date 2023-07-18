import numpy as np
from muse.oracles.push import PushOracle


class MultimodalPushOracle(PushOracle):
    def __init__(self, np_random, min_x_distance, min_y_distance):
        super().__init__(np_random)
        self.min_x_distance = min_x_distance
        self.min_y_distance = min_y_distance

        self.theta_threshold_to_orient = 0.025
        self.theta_threshold_flat_enough = 0.0025
        self.theta_rotation = 0.02
        self.orient_circle_diameter = 0.0225
        self.action_noise_std = 0

    def reset(self):
        super().reset()
        possible_orders = [
            (("cube0", "goal0"), ("cube1", "goal1")),
            (("cube0", "goal1"), ("cube1", "goal0")),
            (("cube1", "goal1"), ("cube0", "goal0")),
            (("cube1", "goal0"), ("cube0", "goal1")),
        ]

        self.goal_order = list(
            possible_orders[self._np_random.choice(list(range(len(possible_orders))))]
        )
        self._obj_name, self._goal_name = self.goal_order.pop(0)

    def get_action(self, obs):
        block_xy = obs[f"{self._obj_name}_pos"][:2]
        goal_xy = obs[f"{self._goal_name}_pos"][:2]
        success = (
            abs(goal_xy[0] - block_xy[0]) < self.min_x_distance
            and abs(goal_xy[1] - block_xy[1]) < self.min_y_distance
        )

        if success:
            if len(self.goal_order) > 0:
                self._obj_name, self._goal_name = self.goal_order.pop(0)
                self.state = "return_to_first_preblock"
            else:
                return None

        action = super().get_action(obs)

        return action
