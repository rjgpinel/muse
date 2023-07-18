import numpy as np

from muse.envs.base import BaseEnv
from muse.script import Script
from muse.envs.utils import random_xy, random_xytheta, create_action_space


class BoxRetrievingEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="box_retrieving_env.xml", **kwargs)
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "grip_open"
        )

        self.box_name = "box"
        self.box_external_color = [0.57647059, 0.65490196, 0.65882353]
        self.box_internal_color = [0.91372549, 0.9372549, 0.9372549]
        self.lid_color = [0.57647059, 0.65490196, 0.65882353]
        self.lid_handle_name = "lid_handle"
        self.lid_handle_color = [0.71372549, 0.69019608, 0.62745098]
        self.box_size = 2 * self.scene.get_geom_size("box_bottom")[1]
        self.lid_height = 2 * self.scene.get_geom_size("lid_handle")[-1]
        self.lid_release_dist = 0.05
        self.target_obj_name = "cube0"
        self.target_obj_size = self.scene.get_geom_size(self.target_obj_name)[-1]
        self.target_site_name = "marker0"
        self.marker_color = [0.365, 0.737, 0.384]
        self.cube_size = self.scene.get_geom_size("cube0")[-1]
        self.marker_size = self.scene.get_geom_size("marker0")[0]

        self.num_obj = 2
        self.obj_list = list(range(self.num_obj))
        self.obj_colors = [(0.74, 0.13, 0.10), (0, 0.48, 0.36)]

        workspace_y_center = (
            (self.workspace[1][1] - self.workspace[0][1]) / 3
        ) + self.workspace[0][1]

        self.box_x, self.box_y = self.scene.get_geom_size("box_bottom")[
            :2
        ]

        self.box_workspace = self.workspace.copy()
        self.box_workspace[1, 1] = workspace_y_center
        self.box_workspace += np.array(
            [[self.box_x, 0.01, 0.01], [-self.box_x, -0.01, -0.01]]
        )

        self.site_workspace = self.workspace.copy()
        self.site_workspace += np.array([[0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]])

        self.gripper_workspace[0, -1] = 0.12

    def reset(self):
        super().reset()

        valid_site = False

        margin = self.box_x + self.marker_size + 0.02
        while not valid_site:
            box_qpos = random_xy(
                1,
                self._np_random,
                self.box_workspace,
                z_pos=0.0001,
                min_dist=0.06,
            )

            self.scene.sim.data.set_mocap_pos(self.box_name, box_qpos[0, :3])
            self.scene.sim.data.set_mocap_quat(self.box_name, box_qpos[0, 3:])

            box_pos = box_qpos[0, :3]

            site_qpos = random_xy(
                1,
                self._np_random,
                self.site_workspace,
                z_pos=0.0001,
                min_dist=0.06,
            )

            site_pos = site_qpos[0, :3]
            if site_pos[0] > box_pos[0] + margin or site_pos[0] < box_pos[0] - margin:
                valid_site = True

        self.scene.sim.data.set_mocap_pos(self.target_site_name, site_qpos[0, :3])
        self.scene.sim.data.set_mocap_quat(self.target_site_name, site_qpos[0, 3:])

        self._np_random.shuffle(self.obj_list)

        for i in range(self.num_obj):
            dist_to_center = 0.05 + self._np_random.uniform(-0.01, 0.01)
            obj_xpos = box_pos.copy()
            if i % 2:
                obj_xpos[0] += int(i / 2 + 1) * dist_to_center
            else:
                obj_xpos[0] -= int(i / 2 + 1) * dist_to_center

            obj_qpos = np.zeros(7)
            obj_qpos[:3] = obj_xpos
            obj_qpos[1] += self._np_random.uniform(-0.015, 0.015)
            obj_qpos[2] = self.cube_size
            obj_qpos[3:] = [1, 0, 0, 0]
            self.scene.set_joint_qpos(f"cube{self.obj_list[i]}_joint", obj_qpos)

        lid_qpos = box_qpos.copy()
        lid_qpos[0, 2] = 0.08
        self.scene.set_joint_qpos("lid_joint", lid_qpos[0])

        if self.scene.modder.domain_randomization:
            for i in range(self.num_obj):
                self.scene.modder.rand_hsv(f"cube{i}", self.obj_colors[i])
            self.scene.modder.rand_hsv(self.target_site_name, self.marker_color)

            # self.lid_color = self._np_random.uniform(0, 255, 3)
            self.box_external_color = self._np_random.uniform(0, 255, 3)
            self.box_internal_color = self._np_random.uniform(0, 255, 3)

            self.scene.modder.rand_hsv("lid_top", self.lid_color)
            self.scene.modder.rand_hsv("lid_handle", self.lid_handle_color)
            self.scene.modder.rand_hsv("box_bottom", self.box_external_color)
            self.scene.modder.rand_hsv(
                "box_separator", self.box_internal_color
            )
        else:
            for i in range(self.num_obj):
                r, g, b = self.obj_colors[i]
                rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
                self.scene.modder.texture_modder.set_rgb(f"cube{i}", rgb)
            r, g, b = self.marker_color
            rgb_marker = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb(self.target_site_name, rgb_marker)

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
            r, g, b = self.box_internal_color
            rgb_int_box = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb(
                "box_separator", rgb_int_box
            )

        self.scene.warmup()
        obs = self.observe()
        return obs

    def script(self):
        script = Script(self.scene, self.gripper_name, velocity_profile="trapezoid")
        lid_handle_pos = self.scene.get_body_pos(self.lid_handle_name)
        target_obj_pos = self.scene.get_body_pos(self.target_obj_name)
        target_site_pos = self.scene.get_site_pos(self.target_site_name)
        box_pos = self.scene.sim.data.get_mocap_pos("box")
        lid_release_pos = np.array(
            [
                box_pos[0],
                lid_handle_pos[1] + (self.box_size + self.lid_release_dist),
                self.lid_height + 0.03,
            ]
        )

        return [
            script.tool_move(lid_handle_pos + [0.0, 0.0, 0.06]),
            script.tool_move(lid_handle_pos),
            script.grip_close(),
            script.tool_move(lid_handle_pos + [0.0, 0.0, 0.06]),
            script.tool_move(
                lid_handle_pos + [0.0, self.box_size + self.lid_release_dist, 0.06]
            ),
            script.tool_move(lid_release_pos),
            script.grip_open(),
            script.tool_move(target_obj_pos + [0.0, 0.0, 0.06]),
            script.tool_move(target_obj_pos),
            script.grip_close(),
            script.tool_move(target_obj_pos + [0.0, 0.0, 0.12]),
            script.tool_move(target_site_pos + [0.0, 0.0, 0.12]),
            script.tool_move(target_site_pos + [0.0, 0.0, self.target_obj_size + 0.02]),
            script.grip_open(),
        ]

    def is_success(self):
        target_obj_pos = self.scene.get_body_pos(self.target_obj_name)
        target_site_pos = self.scene.get_site_pos(self.target_site_name)
        success = (
            np.linalg.norm(target_obj_pos[:2] - target_site_pos[:2]) < 0.01
            and np.abs(target_site_pos[2] - target_obj_pos[2])
            < self.target_obj_size + 0.002
        )
        return success
