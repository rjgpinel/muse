import itertools
import numpy as np

from muse.agent.oracle import OracleAgent
from muse.core import constants


class ScriptAgent(OracleAgent):
    def __init__(self, env, open_gripper_init=False):
        super().__init__(env, open_gripper_init)
        self.scripts = iter(env.script())
        self.steps = itertools.chain(*self.scripts)

    def _compute_steps(self, obs=None):
        script = next(self.scripts, None)
        if script is None:
            self.steps = itertools.chain([])
        else:
            self.steps = itertools.chain(*script)
