import numpy as np

from muse.envs.base import BaseEnv
from muse.script import Script
from muse.envs.utils import create_action_space


class ReachEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="reach_env.xml", **kwargs)
        self.goal_name = "goal0_joint"
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open"
        )
        self.min_goal_distance = 0.015

    def reset(self):
        super().reset(open_gripper=False)
        ee_pos = self.scene.get_ee_pos(self.gripper_name)

        num_goals = 1
        goal_qpos = self._np_random.uniform(
            self.obj_workspace[0], self.obj_workspace[1]
        )

        self.scene.sim.data.set_mocap_pos("goal0", goal_qpos)

        self.scene.warmup()
        obs = self.observe()
        return obs

    def observe(self):
        obs = super().observe()
        obs["goal0_pos"] = self.scene.get_site_pos("goal0")
        return obs

    def script(self):
        script = Script(self.scene, self.gripper_name)
        reach_pos = self.scene.get_site_pos("goal0")
        return [
            script.tool_move(reach_pos),
        ]

    def is_success(self):
        goal_qpos = self.scene.get_site_pos("goal0")
        ee_pos = self.scene.get_ee_pos(self.gripper_name)
        success = np.linalg.norm(goal_qpos - ee_pos) < self.min_goal_distance
        return success
