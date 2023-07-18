import numpy as np

from muse.envs.base import BaseEnv
from muse.script import Script
from muse.envs.utils import random_xy, create_action_space


class StackEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="stack_env.xml", **kwargs)
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open"
        )
        self.num_cubes = 2
        self.count_success = 0
        self.cubes_size = []
        self.cubes_color = [(0, 0.48, 0.36), (0.74, 0.13, 0.10), (0.93, 0.86, 0.16)]

        for i in range(self.num_cubes):
            half_size = self.scene.get_geom_size(f"cube{i}")[2]
            self.cubes_size.append(2 * half_size)

    def reset(self):
        super().reset()

        self.count_success = 0
        valid = False

        while not valid:
            cubes_qpos = random_xy(
                self.num_cubes,
                self._np_random,
                self.obj_workspace,
                z_pos=0.03,
                # min_dist=0.05,
            )
            valid = True
            for i in range(1, self.num_cubes):
                valid = np.abs(cubes_qpos[i][0] - cubes_qpos[i - 1][0]) > 0.06
                if not valid:
                    break

        for i in range(self.num_cubes):
            cube_name = f"cube{i}_joint"
            self.scene.set_joint_qpos(cube_name, cubes_qpos[i])

            if self.scene.modder.domain_randomization:
                self.scene.modder.rand_hsv(f"cube{i}", self.cubes_color[i])
            else:
                r, g, b = self.cubes_color[i]
                rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
                self.scene.modder.texture_modder.set_rgb(f"cube{i}", rgb)

        self.scene.warmup()

        obs = self.observe()
        return obs

    def script(self):
        script = Script(self.scene, self.gripper_name)

        cubes_pos = []
        for i in range(0, self.num_cubes):
            cubes_pos.append(self.scene.get_site_pos(f"cube{i}"))
        cubes_pos = np.stack(cubes_pos)
        cubes_size = self.cubes_size
        place_x, place_y, stack_z = self.scene.get_site_pos("cube0")
        stack_z *= 2

        moves = []
        for pick_pos, placed_cube_size in zip(cubes_pos[1:], cubes_size[:-1]):
            pick_x, pick_y, half_cube_size = pick_pos
            cube_size = 2 * half_cube_size
            stack_z += cube_size
            moves += [
                script.tool_move([pick_x, pick_y, cube_size + 0.06]),
                script.tool_move([pick_x, pick_y, cube_size / 2]),
                script.grip_close(),
                script.tool_move([pick_x, pick_y, stack_z]),
                script.tool_move([place_x, place_y, stack_z]),
                script.grip_open(),
                # script.tool_move([place_x, place_y, stack_z + 0.06]),
            ]

        return moves

    def observe(self):
        obs = super().observe()
        for i in range(0, self.num_cubes):
            obs[f"cube{i}_pos"] = self.scene.get_site_pos(f"cube{i}")
        return obs

    def is_success(self):
        cubes_pos = []
        for i in range(0, self.num_cubes):
            cubes_pos.append(self.scene.get_site_pos(f"cube{i}"))
        cubes_pos = np.stack(cubes_pos)
        cubes_size = self.cubes_size
        place_pos = cubes_pos[0]

        heights = np.cumsum(cubes_size)[1:]
        cubes_xy_mean = np.mean(cubes_pos[1:], axis=0)
        valid_xy = np.linalg.norm(np.subtract(place_pos[:2], cubes_xy_mean[:2])) < 0.02

        valid_z = True
        for cube_pos, cube_size, height in zip(cubes_pos[1:], cubes_size[1:], heights):
            valid_z = valid_z and np.abs(cube_pos[2] + cube_size / 2 - height) < 0.001

        if valid_xy and valid_z:
            self.count_success += 1
        success = self.count_success > 5
        return success
