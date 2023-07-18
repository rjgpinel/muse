import numpy as np

from muse.envs.base import BaseEnv
from muse.script import Script
from muse.envs.utils import random_xy, create_action_space


class BowlEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="bowl_env.xml", **kwargs)
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open"
        )
        self.cube_name = "cube0"
        self.bowl_name = "bowl0"
        self.safe_height = 0.1
        self.cube_size = self.scene.get_geom_size("cube0")[2] * 2

    def reset(self):
        super().reset()

        num_cubes = 1
        cubes_qpos = random_xy(
            num_cubes, self._np_random, self.obj_workspace, z_pos=0.03
        )
        cube_name = "cube0_joint"
        self.scene.set_joint_qpos(cube_name, cubes_qpos[0])

        num_bowls = 1
        valid_bowl_pos = False
        while not valid_bowl_pos:
            bowls_qpos = random_xy(
                num_bowls, self._np_random, self.obj_workspace, z_pos=0.03
            )
            valid_bowl_pos = (
                np.linalg.norm(bowls_qpos[0, :3] - cubes_qpos[0, :3]) > 0.15
            )
        self.scene.sim.data.set_mocap_pos("bowl0", bowls_qpos[0, :3])
        self.scene.sim.data.set_mocap_quat("bowl0", bowls_qpos[0, 3:])

        self.scene.warmup()

        obs = self.observe()
        return obs

    def script(self):
        script = Script(self.scene, self.gripper_name)

        pick_pos = self.scene.get_site_pos("cube0")
        lift_pos = pick_pos.copy()
        lift_pos[-1] = 0.1
        release_pos = self.scene.get_site_pos("bowl0")
        release_pos[-1] = 0.1

        moves = []
        return [
            script.tool_move(lift_pos),
            script.tool_move(pick_pos),
            script.grip_close(),
            script.tool_move(lift_pos),
            script.tool_move(release_pos),
            script.grip_open(),
        ]

    def observe(self):
        obs = super().observe()
        obs["cube0_pos"] = self.scene.get_site_pos("cube0")
        obs["bowl0_pos"] = self.scene.get_site_pos("bowl0")
        return obs

    def is_success(self):
        cube_qpos = self.scene.get_joint_qpos("cube0_joint")
        bowl_qpos = self.scene.get_site_pos("bowl0")

        success = (
            np.linalg.norm(bowl_qpos[:2] - cube_qpos[:2]) < 0.01
            and cube_qpos[2] < self.cube_size
        )

        return success
