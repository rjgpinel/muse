import numpy as np

from muse.core import utils
from muse.envs.base import BaseEnv
from muse.script import Script
from muse.oracles.assembly import NutAssemblyOracle
from muse.envs.utils import random_xy, random_xytheta, create_action_space
from muse.core.utils import (
    muj_to_std_quat,
    quat_to_euler,
    inv_quat,
    euler_to_quat,
    std_to_muj_quat,
    mul_quat,
)


class AssemblyEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(model_path="assembly_env.xml", **kwargs)
        # self.screws_names = ["screw0", "screw1"]
        self.screws_names = ["screw0"]
        self.screws_heights = [
            self.scene.get_geom_size(screw_name)[-1] for screw_name in self.screws_names
        ]
        self.safe_height = max(self.screws_heights) * 2 + 0.03

        # self.nuts_names = ["nut0","nut1"]
        self.nuts_names = ["nut0"]
        self.handles_sizes = [
            self.scene.get_geom_size(f"{nut_name}_handle")[0] * 2
            for nut_name in self.nuts_names
        ]

        workspace_x_center = (
            (self.workspace[1][0] - self.workspace[0][0]) / 2
        ) + self.workspace[0][0]

        self.nuts_workspace = self.workspace.copy()
        self.nuts_workspace[0, 0] = workspace_x_center
        # self.nuts_workspace += np.array([[0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]])
        self.nuts_workspace += np.array([[0.01, 0.075, 0.01], [-0.01, -0.01, -0.01]])

        # print(f"Nut {self.nuts_workspace}")

        self.screws_workspace = self.workspace.copy()
        self.screws_workspace[1, 0] = workspace_x_center
        self.screws_workspace += np.array([[0.01, 0.01, 0.01], [-0.12, -0.01, -0.01]])
        # print(f"Screws {self.screws_workspace}")

        self.gripper_workspace = self.workspace.copy()
        self.gripper_workspace[1, 0] = workspace_x_center

        self.action_space = create_action_space(
            self.scene, "linear_velocity", "theta_velocity", "grip_open"
        )

        self.screws_color = [(0.592, 0.188, 0.365)]
        # self.nuts_color = [(0.412, 0.612, 0.502)]
        self.nuts_color = [(0.2, 0.2, 0.2)]

    def reset(self):
        mocap_target_pos = self._np_random.uniform(
            self.nuts_workspace[0], self.nuts_workspace[1]
        )

        gripper_quat = utils.euler_to_quat((0, 180, 90), degrees=True)
        gripper_quat = utils.std_to_muj_quat(gripper_quat)
        super().reset(mocap_target_pos=mocap_target_pos, mocap_target_quat=gripper_quat)

        num_screws = len(self.screws_names)
        screw_qpos = random_xy(
            num_screws,
            self._np_random,
            self.screws_workspace,
            z_pos=max(self.screws_heights),
            min_dist=0.06,
        )
        for i in range(num_screws):
            screw_qpos[i, 2] = self.screws_heights[i]
            self.scene.sim.data.set_mocap_pos(self.screws_names[i], screw_qpos[i, :3])
            self.scene.sim.data.set_mocap_quat(self.screws_names[i], screw_qpos[i, 3:])
            if self.scene.modder.domain_randomization:
                self.scene.modder.rand_hsv(f"screw{i}", self.screws_color[i])
            else:
                r, g, b = self.screws_color[i]
                rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
                self.scene.modder.texture_modder.set_rgb(f"screw{i}", rgb)

        valid = False
        while not valid:
            num_nuts = len(self.nuts_names)
            nuts_qpos = random_xytheta(
                num_nuts,
                self._np_random,
                self.nuts_workspace,
                z_pos=0.01,
                min_dist=0.07,
                theta_range=[-90, 90],
            )

            valid = True
            for i in range(num_nuts):
                self.scene.set_joint_qpos(f"{self.nuts_names[i]}_joint", nuts_qpos[i])
                handle_pos = self.scene.get_geom_pos(f"{self.nuts_names[i]}_handle")
                if not (
                    self.obj_workspace[0, 0] < handle_pos[0] < self.obj_workspace[1, 0]
                    and self.obj_workspace[0, 1]
                    < handle_pos[1]
                    < self.obj_workspace[1, 1]
                ):
                    valid = False
                    break

                if self.scene.modder.domain_randomization:
                    self.scene.modder.rand_hsv(f"nut{i}_handle", self.nuts_color[i])
                else:
                    r, g, b = self.nuts_color[i]
                    rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
                    self.scene.modder.texture_modder.set_rgb(f"nut{i}_handle", rgb)

        self.scene.warmup()
        obs = self.observe()
        return obs

    def observe(self):
        obs = super().observe()
        for screw_name in self.screws_names:
            obs[f"{screw_name}_pos"] = self.scene.get_body_pos(screw_name)
        for nut_name in self.nuts_names:
            nut_qpos = self.scene.get_joint_qpos(f"{nut_name}_joint")
            obs[f"{nut_name}_pos"] = nut_qpos[:3]
            obs[f"{nut_name}_quat"] = muj_to_std_quat(nut_qpos[3:])
        return obs

    def script(self):
        script = Script(self.scene, self.gripper_name, velocity_profile="trapezoid")

        handle_size = self.handles_sizes[0]

        screw_pos = self.scene.get_body_pos(self.screws_names[0])
        safe_screw_pos = screw_pos.copy()
        safe_screw_pos[-1] = self.safe_height
        nut_handle_pos = self.scene.get_geom_pos(f"{self.nuts_names[0]}_handle")
        # nut_handle_pos[0] -= handle_size / 2
        safe_nut_handle_pos = nut_handle_pos.copy()
        safe_nut_handle_pos[-1] = self.safe_height
        nut_handle_quat = self.scene.get_geom_quat(f"{self.nuts_names[0]}_handle")
        nut_handle_quat = muj_to_std_quat(nut_handle_quat)
        safe_pos = self.scene.get_ee_pos(self.gripper_name)
        safe_pos[-1] = self.safe_height

        nut_hole_pos = self.scene.get_joint_qpos(f"{self.nuts_names[0]}_joint")[:3]
        distance_nut_handle = np.linalg.norm(nut_hole_pos[:2] - nut_handle_pos[:2])

        gripper_quat = muj_to_std_quat(self.scene.get_ee_quat(self.gripper_name))
        final_quat = mul_quat(nut_handle_quat, gripper_quat)

        return [
            script.tool_move(safe_pos),
            script.tool_move(safe_nut_handle_pos, quat=final_quat),
            script.tool_move(nut_handle_pos),
            # script.tool_move(nut_hole_pos),
            script.grip_close(),
            script.tool_move(safe_nut_handle_pos),
            script.tool_move(safe_nut_handle_pos, quat=gripper_quat),
            script.tool_move(safe_screw_pos + [distance_nut_handle, 0.0, 0.0]),
            script.tool_move(screw_pos + [distance_nut_handle, 0.0, 0.0]),
            script.grip_open(),
        ]

    def is_success(self):
        nut_qpos = self.scene.get_joint_qpos(f"{self.nuts_names[0]}_joint")
        nut_pos = nut_qpos[:3]
        screw_pos = self.scene.get_body_pos(self.screws_names[0])

        success = (
            np.linalg.norm(nut_pos[:2] - screw_pos[:2]) < 0.0175
            and nut_pos[-1] < self.screws_heights[-1] / 2
        )
        return success

    def step(self, action):
        new_action = action.copy()
        new_action["angular_velocity"] = np.zeros(3)
        new_action["angular_velocity"][-1] = new_action.pop("theta_velocity", 0.0)
        return super().step(new_action)

    def oracle(self):
        return NutAssemblyOracle(
            self,
        )
