import numpy as np
import math
from muse.envs.base import BaseEnv
from muse.script import Script
from muse.envs.utils import random_xy, create_action_space


class PushAndPickEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="push_and_pick_env.xml", **kwargs)
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open"
        )
        # Num obstacles per cube
        half_size = self.scene.get_geom_size("cube0")[2]
        self.cube_size = 2 * half_size
        self.num_obstacles = 3
        self.obstacles_size = []
        for i in range(self.num_obstacles):
            half_size = self.scene.get_geom_size(f"obstacle{i}")[2]
            self.obstacles_size.append(2 * half_size)

        self.cube_color = [0.74, 0.13, 0.10]
        self.obstacles_color = (
            np.array([[0, 0.26, 0.70], [0, 0.48, 0.36], [1, 1, 0.29]]) * 255
        )

        self.obj_workspace = self.workspace.copy()
        self.obj_workspace += np.array(
            [
                [sum(self.obstacles_size) / 4, sum(self.obstacles_size) / 3, 0.01],
                [
                    -sum(self.obstacles_size) / 4 / 3,
                    -sum(self.obstacles_size) / 3,
                    -0.01,
                ],
            ]
        )

        self.gripper_workspace[0, 2] = 0.07

    def reset(self):
        super().reset()

        num_cubes = 1
        cubes_qpos = random_xy(
            num_cubes, self._np_random, self.obj_workspace, z_pos=self.cube_size
        )

        cube_qpos = cubes_qpos[0]
        self.scene.set_joint_qpos("cube0_joint", cube_qpos)
        if self.scene.modder.domain_randomization:
            self.scene.modder.rand_hsv(f"cube0", self.cube_color)
        else:
            r, g, b = self.cube_color
            rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb("cube0", rgb)

        left_space = self.cube_size / 2
        right_space = self.cube_size / 2
        back_space = self.cube_size / 2
        front_space = self.cube_size / 2
        for i in range(self.num_obstacles):
            obstacle_qpos = cube_qpos.copy()
            if i < self.num_obstacles / 2:
                obstacle_size = self.obstacles_size[i]
                if i % 2:
                    obstacle_qpos[1] += right_space + obstacle_size / 2
                    right_space += obstacle_size / 2
                else:
                    obstacle_qpos[1] -= left_space + obstacle_size / 2
                    left_space += obstacle_size / 2
            else:
                if i % 2:
                    obstacle_qpos[0] -= front_space + obstacle_size / 2
                    front_space += obstacle_size / 2
                else:
                    obstacle_qpos[0] += back_space + obstacle_size / 2
                    back_space += obstacle_size / 2

            self.scene.set_joint_qpos(f"obstacle{i}_joint", obstacle_qpos)

            rgb = self.obstacles_color[i]
            self.scene.modder.texture_modder.set_rgb(f"obstacle{i}", rgb)
            rgb = self._np_random.uniform(0, 255, 3)
            r, g, b = rgb
            self.scene.modder.rand_hsv(f"obstacle{i}", (r, g, b))

        self.scene.warmup()

        obs = self.observe()
        return obs

    def script(self):
        script = Script(self.scene, self.gripper_name)

        pick_pos = self.scene.get_site_pos("cube0")

        moves = [script.grip_close()]
        safe_height = 0.06
        push_safe_height = 0.06
        for i in range(math.ceil(self.num_obstacles / 2)):
            obstacle_pos = self.scene.get_site_pos(f"obstacle{i}")
            obstacle_size = self.obstacles_size[i]
            moves += [
                script.tool_move(
                    obstacle_pos + [-obstacle_size, 0.0, push_safe_height]
                ),
                script.tool_move(obstacle_pos + [-obstacle_size, 0.0, 0.0]),
                script.tool_move(
                    obstacle_pos + [self.cube_size / 2 + obstacle_size / 2, 0.0, 0.0]
                ),
                script.tool_move(obstacle_pos - [obstacle_size, 0.0, 0.0]),
            ]
            push_safe_height = 0.0

        moves += [
            script.grip_open(),
            script.tool_move(pick_pos + [0, 0, safe_height]),
            script.tool_move(pick_pos + [0, 0, 0.0]),
            script.grip_close(),
            script.tool_move(pick_pos + [0, 0, safe_height]),
        ]

        return moves

    def observe(self):
        obs = super().observe()
        obs["cube0_pos"] = self.scene.get_site_pos("cube0")
        for i in range(self.num_obstacles):
            obs[f"obstacle{i}"] = self.scene.get_site_pos(f"obstacle{i}")
        return obs

    def is_success(self):
        cube_qpos = self.scene.get_joint_qpos("cube0_joint")
        success = cube_qpos[2] > 0.04
        return success
