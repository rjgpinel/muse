import numpy as np

from muse.agent.script import ScriptAgent


class NutAssemblyOracle(ScriptAgent):
    def __init__(
        self,
        env,
    ):
        super().__init__(env, True)

    def get_action(self, obs):
        action = super().get_action(obs)
        if action is not None:
            angular_velocity = action.pop("angular_velocity", np.zeros(3))
            theta_velocity = np.array([angular_velocity[-1]])
            action["theta_velocity"] = theta_velocity
        return action
