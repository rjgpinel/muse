import copy
import numpy as np

from muse.envs.base import BaseEnv
from muse.script import Script
from muse.envs.utils import random_xy, random_xytheta, create_action_space

COLORS = {
    "maroon": (0.5, 0.0, 0.0),
    "green": (0.0, 0.5, 0.0),
    "navy": (0.0, 0.0, 0.5),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "orange": (1.0, 0.5, 0.0),
    "rose": (1.0, 0.0, 0.5),
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
}


class BoxButtonsEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="box_buttons_env.xml", **kwargs)
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open"
        )

        self.box_name = "box"
        self.box_external_color = [0.57647059, 0.65490196, 0.65882353]
        self.box_internal_color = [0.91372549, 0.9372549, 0.9372549]
        self.lid_color = [0.57647059, 0.65490196, 0.65882353]
        self.lid_handle_name = "lid_handle"
        self.lid_handle_color = [0.71372549, 0.69019608, 0.62745098]
        self.box_halfsize = self.scene.get_geom_size("box_bottom")[:2]
        self.box_xy = self.scene.get_geom_pos("box_bottom")[:2]
        self.lid_height = 2 * self.scene.get_geom_size("lid_handle")[-1]

        self.obj_name = "cube0"
        self.obj_size = self.scene.get_geom_size(self.obj_name)[-1]

        self.obj_workspace = np.array(
            [
                self.box_xy - self.box_halfsize + self.obj_size * 1.5,
                self.box_xy + self.box_halfsize - self.obj_size * 1.5,
            ]
        )

        self.num_buttons = 3
        self.num_distractors = self.num_buttons - 1
        self.buttons_xpos = -0.6
        self.buttons_ypos = np.linspace(
            self.workspace[0][1], self.workspace[1][1], num=self.num_buttons
        )

        self.buttons_pos = [
            np.array([self.buttons_xpos, self.buttons_ypos, 0.01])
            for y in self.buttons_ypos
        ]
        self.target_id = 0

        self.button_height_offset = 0.06
        self.gripper_workspace[0, -1] = 0.12

        self.buttons_pressure = [list() for _ in range(self.num_buttons)]
        self.buttons_state = [False] * self.num_buttons
        self.min_value_touched = 150

    def reset(self):
        super().reset()

        obj_xy = self._np_random.uniform(
            low=self.obj_workspace[0], high=self.obj_workspace[1]
        )

        obj_pos = np.concatenate([obj_xy, [self.obj_size], [1, 0, 0, 0]], axis=0)

        self.scene.set_joint_qpos(f"{self.obj_name}_joint", obj_pos)

        distractors_id = list(range(self.num_buttons))

        self.target_id = self._np_random.choice(distractors_id)
        distractors_id.remove(self.target_id)
        target_color_name = self._np_random.choice(list(COLORS.keys()))
        target_color = COLORS[target_color_name]
        r, g, b = target_color
        rgb_button = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)

        if self.scene.modder.domain_randomization:
            self.scene.modder.rand_hsv(f"button{self.target_id}", (r, g, b))
        else:
            self.scene.modder.texture_modder.set_rgb(
                f"button{self.target_id}", rgb_button
            )

        distractors_colors = copy.deepcopy(COLORS)
        distractors_colors.pop(target_color_name)
        distractors_colors = self._np_random.choice(
            list(distractors_colors.keys()), size=self.num_distractors, replace=False
        )

        for i in range(self.num_distractors):
            color_name = distractors_colors[i]
            color = COLORS[color_name]
            r, g, b = color
            rgb_button = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            button_id = distractors_id[i]
            if self.scene.modder.domain_randomization:
                self.scene.modder.rand_hsv(f"button{button_id}", (r, g, b))
            else:
                self.scene.modder.texture_modder.set_rgb(
                    f"button{button_id}", rgb_button
                )

        if self.scene.modder.domain_randomization:
            self.scene.modder.rand_hsv(self.obj_name, target_color)
            self.scene.modder.rand_hsv("lid_top", self.lid_color)
            self.scene.modder.rand_hsv("lid_handle", self.lid_handle_color)
            self.scene.modder.rand_hsv("box_bottom", self.box_external_color)
        else:
            r, g, b = target_color
            rgb_obj = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb(self.obj_name, rgb_obj)
            r, g, b = self.lid_color
            rgb_lid = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb("lid_top", rgb_lid)
            r, g, b = self.lid_handle_color
            rgb_lid_handle = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb("lid_handle", rgb_lid_handle)
            r, g, b = self.box_external_color
            rgb_ext_box = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb(
                "box_bottom", rgb_ext_box
            )

        self.buttons_pressure = [list() for _ in range(self.num_buttons)]

        self.scene.warmup()
        obs = self.observe()
        return obs

    def script(self):
        script = Script(self.scene, self.gripper_name, velocity_profile="trapezoid")
        lid_handle_pos = self.scene.get_body_pos(self.lid_handle_name)
        button_pos = self.scene.get_body_pos(f"button{self.target_id}")
        return [
            script.tool_move(lid_handle_pos + [0.0, 0.0, 0.06]),
            script.tool_move(lid_handle_pos),
            script.grip_close(),
            script.tool_move(lid_handle_pos + [0.0, 0.0, 0.08]),
            script.pause(),
            script.tool_move(lid_handle_pos + [0.0, 0.0, 0.01]),
            script.grip_open(),
            script.tool_move(lid_handle_pos + [-0.05, 0.0, 0.06]),
            script.grip_close(),
            script.tool_move(button_pos + [0.0, 0.0, self.button_height_offset]),
            script.tool_move(button_pos + [0.0, 0.0, 0.01]),
        ]

    def step(self, action):
        for i in range(self.num_buttons):
            button_pressure = self.scene.sim.data.get_sensor(f"button{i}_sensor")
            self.buttons_pressure[i].append(button_pressure)
            pushed = np.mean(self.buttons_pressure[i][-5:]) > self.min_value_touched
            if pushed and self.buttons_state[i] is False:
                self.buttons_state[i] = True
            elif not pushed:
                self.buttons_state[i] = False
        return super().step(action)

    def is_success(self):
        return self.buttons_state[self.target_id] and not np.any(
            np.concatenate(
                [
                    self.buttons_state[: self.target_id],
                    self.buttons_state[self.target_id + 1 :],
                ]
            )
        )
