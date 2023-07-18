import copy
import json
import numpy as np
import xml.etree.ElementTree as ET

from muse.envs.base import BaseEnv
from muse.script import Script
from muse.envs.utils import random_xy, create_action_space
from muse.core import utils


COLORS = {
    "maroon": (0.5, 0.0, 0.0),
    "green": (0.0, 0.5, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "navy": (0.0, 0.0, 0.5),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "silver": (0.75, 0.75, 0.75),
    "gray": (0.5, 0.5, 0.5),
    "orange": (1.0, 0.5, 0.0),
    "olive": (0.5, 0.5, 0.0),
    "purple": (0.5, 0.0, 0.5),
    "teal": (0, 0.5, 0.5),
    "azure": (0.0, 0.5, 1.0),
    "violet": (0.5, 0.0, 1.0),
    "rose": (1.0, 0.0, 0.5),
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
}


class PushButtonsEnv(BaseEnv):
    def __init__(self, variant_id=0, pushing_order=[0, 1, 0, 2], **kwargs):
        super().__init__(model_path="push_buttons_env.xml", **kwargs)
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open"
        )

        with open(str(utils.assets_dir() / "push_buttons.json"), "r") as f:
            self.variants = json.load(f)

        self.variant_id = variant_id

        self.variant = self.variants[self.variant_id]
        self.num_buttons = len(self.variant)
        self.height_offset = 0.06

        self.buttons_state = [False] * self.num_buttons
        self.buttons_pressure = [list() for _ in range(self.num_buttons)]
        self.min_value_touched = 150
        self.step_id = 0
        self.num_distractors = 3 - self.num_buttons

        self.pushing_order = pushing_order
        self.pushed_buttons = []
        self.pushed_timesteps = []

        self.timestep = -1

        # print("num buttons", self.num_buttons)
        # print("num_distractors", self.num_distractors)

    def seed(self, seed=None):
        if seed is not None:
            seed = seed * int(2e4)
            seed += self.variant_id * int(2e2)
        return super().seed(seed % int(2 ** 32))

    def reset(self):
        super().reset()
        self.step_id = 0

        assert (
            self.num_buttons + self.num_distractors == 3
        ), "Distractors + Buttons number should be 3"

        buttons_qpos = random_xy(
            self.num_buttons + self.num_distractors,
            self._np_random,
            self.obj_workspace,
            z_pos=0.01,
            min_dist=0.055,
        )

        self.scene.modder.texture_modder.whiten_materials(
            [f"button{i}" for i in range(self.num_buttons)]
        )

        distractors_colors = copy.deepcopy(COLORS)

        for i, button_info in enumerate(self.variant):
            color_name, button_color = button_info
            distractors_colors.pop(color_name)
            # print("Button color: ", color_name)
            button_name = f"button{i}_box"
            self.scene.sim.data.set_mocap_pos(button_name, buttons_qpos[i, :3])
            self.scene.sim.data.set_mocap_quat(button_name, buttons_qpos[i, 3:])
            r, g, b = button_color
            button_color = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb(f"button{i}", button_color)
            if self.scene.modder.domain_randomization:
                self.scene.modder.rand_hsv(f"button{i}", (r, g, b))

        self.buttons_state = [False] * self.num_buttons
        self.pushed_buttons = []
        self.buttons_pressure = [list() for _ in range(self.num_buttons)]
        self.timestep = -1

        for i in range(self.num_distractors):
            color_id = self._np_random.choice(list(range(len(distractors_colors))))
            color_name = list(distractors_colors.keys())[color_id]
            button_color = distractors_colors[color_name]
            distractors_colors.pop(color_name)
            button_id = i + self.num_buttons
            # print("Disctractor color: ", color_name)
            button_name = f"button{button_id}_box"
            self.scene.sim.data.set_mocap_pos(button_name, buttons_qpos[button_id, :3])
            self.scene.sim.data.set_mocap_quat(button_name, buttons_qpos[button_id, 3:])
            r, g, b = button_color
            button_color = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb(f"button{button_id}", button_color)
            if self.scene.modder.domain_randomization:
                self.scene.modder.rand_hsv(f"button{button_id}", (r, g, b))

        self.scene.warmup()

        obs = self.observe()
        return obs

    def script(self):
        script = Script(self.scene, self.gripper_name)
        moves = [script.grip_close()]
        for i in self.pushing_order:
            pos_button = self.scene.get_body_pos(f"button{i}")
            moves += [
                script.tool_move(pos_button + [0.0, 0.0, self.height_offset]),
                script.tool_move(pos_button + [0.0, 0.0, 0.01]),
                script.tool_move(pos_button + [0.0, 0.0, 0.03]),
            ]

        return moves

    def observe(self):
        obs = super().observe()
        for i in range(self.num_buttons):
            pos_button = self.scene.get_body_pos(f"button{i}")
            obs[f"button{i}_pos"] = pos_button
        return obs

    def step(self, action):
        self.timestep += 1
        # print(self.scene.sim.data.sensordata)
        # print(self.buttons_pressure[:][-5:])
        self.step_id += 1
        for i in range(self.num_buttons):
            button_pressure = self.scene.sim.data.get_sensor(f"button{i}_sensor")
            self.buttons_pressure[i].append(button_pressure)
            # print(self.buttons_pressure[i][-5:])
            # print(np.mean(self.buttons_pressure[i][-5:]))
            pushed = np.mean(self.buttons_pressure[i][-5:]) > self.min_value_touched
            if pushed and self.buttons_state[i] is False:
                self.buttons_state[i] = True
                self.pushed_buttons.append(i)
                self.pushed_timesteps.append(self.timestep)
                # print(f"Button {i} pushed - Step id: {self.step_id}")
            elif not pushed:
                self.buttons_state[i] = False
        return super().step(action)

    def is_success(self):
        # print(self.pushed_buttons)
        if len(self.pushed_buttons) >= len(self.pushing_order):
            success = (
                self.pushed_buttons[-len(self.pushing_order) :] == self.pushing_order
            )
            # if success:
            # print(f"Steps: {self.pushed_timesteps[-len(self.pushing_order) :]}")
            return success
        else:
            return False
