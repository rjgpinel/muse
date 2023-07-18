import numpy as np

from muse.envs.base import BaseEnv
from muse.script import Script
from muse.envs.utils import random_xy, create_action_space


class StackButtonsEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="stack_buttons_env.xml", **kwargs)
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open"
        )

        self.obj_minx_dist = 0.06

        self.num_cubes = 2
        self.count_success = 0
        self.cubes_size = []
        self.cubes_color = [(0, 0.48, 0.36), (0.74, 0.13, 0.10), (0.93, 0.86, 0.16)]
        for i in range(self.num_cubes):
            half_size = self.scene.get_geom_size(f"cube{i}")[2]
            self.cubes_size.append(2 * half_size)

        self.button_color = (0.0, 0.0, 0.0)
        self.button_pushed = False
        self.button_pressure = []
        self.button_timestep = -1
        self.button_height_offset = 0.06
        self.timestep = -1
        self.stack_success_timestep = -1
        self.min_value_pressed = 150

    def reset(self):
        super().reset()

        self.count_success = 0
        valid = False
        while not valid:
            objects_qpos = random_xy(
                self.num_cubes + 1,  # num_cubes + 1 button
                self._np_random,
                self.obj_workspace,
                z_pos=0.03,
                min_dist=0.04,
            )
            cubes_qpos = objects_qpos[: self.num_cubes]
            valid = True
            for i in range(1, self.num_cubes + 1):
                valid = (
                    np.abs(objects_qpos[i][0] - objects_qpos[i - 1][0])
                    > self.obj_minx_dist
                )
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

        self.button_pushed = False
        self.button_timestep = -1
        self.timestep = -1
        self.button_pressure = []
        self.stack_success_timestep = -1

        button_name = "button0_box"
        button_qpos = objects_qpos[-1]
        button_qpos[2] = 0.01  # button height
        self.scene.sim.data.set_mocap_pos(button_name, button_qpos[:3])
        self.scene.sim.data.set_mocap_quat(button_name, button_qpos[3:])

        if self.scene.modder.domain_randomization:
            self.scene.modder.rand_hsv("button0", self.button_color)
        else:
            r, g, b = self.button_color
            rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb("button0", rgb)

        self.scene.warmup()

        obs = self.observe()
        return obs

    def step(self, action):
        self.timestep += 1
        button_pressure = self.scene.sim.data.get_sensor(f"button0_sensor")
        self.button_pressure.append(button_pressure)
        pushed = np.mean(self.button_pressure[-5:]) > self.min_value_pressed
        if pushed and self.button_pushed is False:
            self.button_pushed = True
            self.button_timestep = self.timestep
        elif not pushed:
            self.button_pushed = False
        return super().step(action)

    def script(self):
        script = Script(self.scene, self.gripper_name)

        button_pos = self.scene.get_body_pos(f"button0")

        cubes_pos = []
        for i in range(0, self.num_cubes):
            cubes_pos.append(self.scene.get_site_pos(f"cube{i}"))
        cubes_pos = np.stack(cubes_pos)
        cubes_size = self.cubes_size
        place_x, place_y, stack_z = self.scene.get_site_pos("cube0")
        stack_z *= 2

        moves = [
            script.tool_move(button_pos + [0.0, 0.0, self.button_height_offset]),
            script.grip_close(),
            script.tool_move(button_pos + [0.0, 0.0, 0.01]),
            script.tool_move(button_pos + [0.0, 0.0, self.button_height_offset]),
            script.grip_open(),
        ]

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

        if self.count_success > 5:
            self.stack_success_timestep = self.timestep

        success = (
            self.count_success > 5
            and self.stack_success_timestep > self.button_timestep > -1
        )
        return success
