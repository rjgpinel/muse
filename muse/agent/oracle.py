import itertools
import numpy as np

from muse.core import constants


class OracleAgent:
    def __init__(self, env, open_gripper_init=False):
        self.action_space = env.action_space
        self.last_gripper_action = (
            constants.GRIP_OPEN if open_gripper_init else constants.GRIP_CLOSE
        )
        self.steps = iter([])

    def _compute_steps(self, obs=None):
        self.steps = itertools.chain([])

    def get_action(self, obs=None):
        action = next(self.steps, None)
        if action is None:
            self._compute_steps(obs)
            action = next(self.steps, None)

        if action is None:
            return None

        if "grip_open" in self.action_space.keys():
            if "grip_open" in action:
                self.last_gripper_action = action["grip_open"]
            # define current gripper action as the last executed one if grip_open
            # is not in the action keys
            action["grip_open"] = self.last_gripper_action

        for k, v in self.action_space.items():
            if k not in action and k != "grip_open":
                action[k] = np.zeros(v.shape)

        return action
