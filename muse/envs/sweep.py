import numpy as np
import xml.etree.ElementTree as ET

from muse.core import utils
from copy import copy
from muse.envs.base import BaseEnv
from muse.oracles.sweep import SweepOracle
from muse.envs.utils import random_xy, create_action_space


class SweepEnv(BaseEnv):
    def __init__(self, **kwargs):
        # create_new_template = True
        create_new_template = False
        self.num_piles = 15
        self.pile_size = 0.0075
        piles_spacing = 0.05
        self.default_pile_color = np.array([0.859, 0.329, 0.376])
        # self.default_pile_color = np.array([0.541, 0.075, 0.067])
        if create_new_template:
            tree = ET.parse(str(utils.assets_dir() / "sweep_env_template.xml"))
            root = tree.getroot()
            worldbody = ET.SubElement(root, "worldbody")
            asset = ET.SubElement(root, "asset")

            ET.SubElement(
                asset,
                "texture",
                attrib=dict(
                    builtin="flat",
                    type="2d",
                    name=f"pile_tex",
                    height="32",
                    width="32",
                ),
            )
            ET.SubElement(
                asset,
                "material",
                attrib=dict(
                    name=f"pile_mat",
                    specular="0.5",
                    shininess="0.2",
                    reflectance="0",
                    texture=f"pile_tex",
                ),
            )
            for i in range(self.num_piles):
                pile_body = ET.SubElement(
                    worldbody,
                    "body",
                    attrib=dict(
                        name=f"pile{i}",
                        pos="0 0 0",
                    ),
                )
                ET.SubElement(
                    pile_body,
                    "joint",
                    attrib=dict(
                        name=f"pile{i}_joint_x",
                        pos="0 0 0",
                        axis="1 0 0",
                        type="slide",
                        group="3",
                        damping="0",
                    ),
                )
                ET.SubElement(
                    pile_body,
                    "joint",
                    attrib=dict(
                        name=f"pile{i}_joint_y",
                        pos="0 0 0",
                        axis="0 1 0",
                        type="slide",
                        group="3",
                        damping="0",
                    ),
                )
                ET.SubElement(
                    pile_body,
                    "joint",
                    attrib=dict(
                        name=f"pile{i}_joint_z",
                        pos="0 0 0",
                        axis="0 0 1",
                        type="slide",
                        group="3",
                        damping="0",
                    ),
                )
                ET.SubElement(
                    pile_body,
                    "geom",
                    attrib=dict(
                        name=f"pile{i}",
                        size="0.0075 0.0075 0.0075",
                        type="box",
                        material=f"pile_mat",
                        mass="0.01",
                    ),
                )

                ET.SubElement(
                    pile_body,
                    "site",
                    attrib=dict(name=f"pile{i}", size="0.005 0.005 0.005", type="box"),
                )
                tree.write(str(utils.assets_dir() / "sweep_env.xml"))

        super().__init__(model_path="sweep_env.xml", **kwargs)
        self.action_space = create_action_space(
            self.scene, "linear_velocity", "theta_velocity"
        )
        self.broom_name = "broom_joint"
        self.marker_name = "marker0"
        self.marker_halfsize = self.scene.get_geom_size("marker0")[:2]
        self.marker_color = [0.365, 0.737, 0.384]
        self.broom_size = self.scene.get_geom_size("broom")[-1]
        self.safe_height = self.broom_size * 2

        workspace_x_center = (
            (self.workspace[1][0] - self.workspace[0][0]) / 2
        ) + self.workspace[0][0]

        self.goal_workspace = self.workspace.copy()
        self.goal_workspace[0, 0] = workspace_x_center
        self.goal_workspace += np.array([[0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]])

        self.piles_workspace = self.workspace.copy()
        self.piles_workspace[1, 0] = workspace_x_center
        self.piles_workspace += np.array([[0.05, 0.05, 0.01], [-0.01, -0.05, -0.01]])

        # self.gripper_workspace = self.piles_workspace.copy()
        # self.gripper_workspace[0, 0] = self.safe_height

    def reset(self):
        mocap_target_pos = self._np_random.uniform(
            self.gripper_workspace[0], self.gripper_workspace[1]
        )
        # print(f"Pos {mocap_target_pos}")
        gripper_quat = utils.euler_to_quat((0, 180, 180), degrees=True)
        # gripper_quat = utils.euler_to_quat((0, 180, 0), degrees=True)
        gripper_quat = utils.std_to_muj_quat(gripper_quat)
        super().reset(
            open_gripper=False,
            mocap_target_pos=mocap_target_pos,
            mocap_target_quat=gripper_quat,
        )

        piles_qpos = random_xy(
            self.num_piles,
            self._np_random,
            self.piles_workspace,
            z_pos=self.pile_size,
            min_dist=0.002,
        )

        self.scene.sim.data.qpos[-self.num_piles * 3 :] = piles_qpos[:, :3].flatten()

        gripper_pos = self.scene.get_ee_pos(self.gripper_name)
        broom_pos = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        broom_pos[:3] = gripper_pos
        self.scene.set_joint_qpos(self.broom_name, broom_pos)
        if self.scene.modder.domain_randomization:
            for i in range(self.num_piles):
                self.scene.modder.rand_hsv(f"pile{i}", self.default_pile_color)
            self.scene.modder.rand_hsv(self.marker_name, self.marker_color)
        else:
            for i in range(self.num_piles):
                r, g, b = self.default_pile_color
                rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
                self.scene.modder.texture_modder.set_rgb(f"pile{i}", rgb)
            r, g, b = self.marker_color
            rgb_marker = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
            self.scene.modder.texture_modder.set_rgb(self.marker_name, rgb_marker)

        marker_qpos = random_xy(1, self._np_random, self.goal_workspace, z_pos=0.0001)
        self.scene.sim.data.set_mocap_pos(self.marker_name, marker_qpos[0, :3])
        self.scene.sim.data.set_mocap_quat(self.marker_name, marker_qpos[0, 3:])

        self.scene.warmup()
        obs = self.observe()
        return obs

    def observe(self):
        obs = super().observe()
        obs[f"{self.marker_name}_pos"] = self.scene.get_site_pos(self.marker_name)
        for i in range(self.num_piles):
            obs[f"pile{i}_pos"] = self.scene.get_geom_pos(f"pile{i}")
            obs[f"pile{i}_goal_dist"] = np.linalg.norm(
                obs[f"{self.marker_name}_pos"][:2]
                - self.scene.get_geom_pos(f"pile{i}")[:2]
            )
        return obs

    def oracle(self):
        return SweepOracle(
            self,
            self._np_random,
            self.gripper_name,
            self.num_piles,
            piles_size=self.pile_size,
            broom_size=self.broom_size,
            safe_height=self.safe_height,
        )

    def is_success(self):
        success = True
        marker_pos = self.scene.get_site_pos(self.marker_name)
        margin_success = 0.005
        for i in range(self.num_piles):
            pile_pos = self.scene.get_geom_pos(f"pile{i}")
            if (
                np.abs(marker_pos[0] - pile_pos[0])
                > self.marker_halfsize[0] + margin_success
                or np.abs(marker_pos[1] - pile_pos[1])
                > self.marker_halfsize[1] + margin_success
            ):
                success = False
                break
        return success

    def step(self, action):
        new_action = action.copy()
        new_action["angular_velocity"] = np.zeros(3)
        new_action["angular_velocity"][-1] = new_action.pop("theta_velocity", 0.0)
        return super().step(new_action)
