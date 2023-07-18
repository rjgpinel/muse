import numpy as np

from muse.envs.base import BaseEnv
from muse.script import Script
from muse.envs.utils import random_xy, create_action_space


class PickEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="pick_env.xml", **kwargs)
        self.cube_name = "cube0_joint"
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open"
        )
        self.num_distractors = 2
        self.cubes_color = [
            (0.74, 0.13, 0.10),
            (0, 0.48, 0.36),
            (0.93, 0.86, 0.16),
        ]

    def reset(self):
        super().reset()

        valid = False
        num_cubes = 1 + self.num_distractors

        while not valid:
            cubes_qpos = random_xy(
                num_cubes,
                self._np_random,
                self.obj_workspace,
                z_pos=0.03,
                min_dist=0.06,
            )
            valid = True
            for cube_i in range(num_cubes):
                for cube_j in range(num_cubes):
                    if (
                        np.linalg.norm(
                            cubes_qpos[cube_i][:3]
                            - self.scene.get_site_pos(f"cube{cube_j}")
                        )
                        <= 0.06
                    ):
                        valid = False

            if valid:
                for cube_i in range(num_cubes):
                    self.scene.set_joint_qpos(f"cube{cube_i}_joint", cubes_qpos[cube_i])

                    if self.scene.modder.domain_randomization:
                        self.scene.modder.rand_hsv(
                            f"cube{cube_i}", self.cubes_color[cube_i]
                        )
                    else:
                        r, g, b = self.cubes_color[cube_i]
                        rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
                        self.scene.modder.texture_modder.set_rgb(f"cube{cube_i}", rgb)

        self.scene.warmup()

        obs = self.observe()
        return obs

    def observe(self):
        obs = super().observe()
        num_cubes = 1 + self.num_distractors
        for cube_i in range(num_cubes):
            obs[f"cube{cube_i}_pos"] = self.scene.get_site_pos(f"cube{cube_i}")
        return obs

    def script(self):
        script = Script(self.scene, self.gripper_name)
        pick_pos = self.scene.get_site_pos("cube0")
        return [
            script.tool_move(pick_pos + [0, 0, 0.06]),
            script.tool_move(pick_pos + [0, 0, 0.0]),
            script.grip_close(),
            script.tool_move(pick_pos + [0, 0, 0.06]),
        ]

    def is_success(self):
        cube_qpos = self.scene.get_joint_qpos(self.cube_name)
        success = cube_qpos[2] > 0.04
        return success
