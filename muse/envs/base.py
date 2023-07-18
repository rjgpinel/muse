import numpy as np
import gym
from gym.utils import seeding

from muse.core.scene import Scene
from muse.core.utils import muj_to_std_quat, quat_to_euler
from muse.core import constants
from muse.agent.script import ScriptAgent
from muse.envs import utils


WORKSPACE = np.array([[-0.695, -0.175, 0.00], [-0.295, 0.175, 0.2]])
GRIPPER_HEIGHT_INIT = np.array([0.06, 0.10])


class BaseEnv(gym.Env):
    def __init__(
        self,
        model_path="",
        viewer=False,
        cam_resolution=constants.RENDER_RESOLUTION,
        cam_render=True,
        # cam_list=["bravo_camera", "charlie_camera", "left_mounted_camera"],
        cam_list=["bravo_camera", "charlie_camera"],
        segmentation_masks=False,
        cam_crop=True,
        domain_randomization=False,
        num_textures=None,
        num_distractors=0,
        textures_pack="bop",
        cam_xyz_noise=0.0,
        light_setup="default",
        gripper_name="left_gripper",
    ):
        self.scene = Scene(
            model_path,
            viewer,
            domain_randomization,
            num_textures=num_textures,
            textures_pack=textures_pack,
            cam_xyz_noise=cam_xyz_noise,
            light_setup=light_setup,
        )

        self.cam_render = cam_render
        self.cam_list = cam_list
        self.cam_resolution = cam_resolution
        self.cam_crop = cam_crop
        self.segmentation_masks = segmentation_masks

        # workspace
        self.workspace = WORKSPACE

        self.gripper_name = gripper_name
        self.gripper_workspace = self.workspace.copy()
        self.gripper_workspace[:, 2] = GRIPPER_HEIGHT_INIT

        self.obj_workspace = self.workspace + np.array(
            [[0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]]
        )

        self._np_random = np.random

        # TODO: Define in IBC repo, maybe? - or move all state data here, including sin-cos
        self.raw_state_space = ["gripper_pos", "gripper_theta"]
        self.state_dim = 5

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        self._np_random = np_random
        self._seed = seed
        return seed

    def reset(self, mocap_target_pos=None, mocap_target_quat=None, open_gripper=True):
        if mocap_target_pos is None:
            # mocap_target_pos = self.workspace.mean(0)
            mocap_target_pos = self._np_random.uniform(
                self.gripper_workspace[0], self.gripper_workspace[1]
            )

        self.open_gripper_init = open_gripper

        mocap_target_pos_dict = dict()
        mocap_target_pos_dict[self.gripper_name] = mocap_target_pos
        if mocap_target_quat is None:
            mocap_target_quat_dict = None
        else:
            mocap_target_quat_dict = dict()
            mocap_target_quat_dict[self.gripper_name] = mocap_target_quat
        open_gripper_dict = dict()
        open_gripper_dict[self.gripper_name] = open_gripper

        self.scene.reset(
            mocap_pos=mocap_target_pos_dict,
            mocap_quat=mocap_target_quat_dict,
            open_gripper=open_gripper_dict,
            workspace=self.workspace,
        )

    def render(self):
        self.scene.render()

    def observe(self):
        obs = dict()
        obs["gripper_pos"] = self.scene.get_ee_pos(self.gripper_name)
        obs["gripper_quat"] = muj_to_std_quat(self.scene.get_ee_quat(self.gripper_name))
        obs["gripper_joint_qpos"] = self.scene.get_gripper_qpos(self.gripper_name)
        obs["gripper_theta"] = quat_to_euler(obs["gripper_quat"], False)[-1]

        if self.cam_render:
            w, h = self.cam_resolution
            for cam_name in self.cam_list:
                im = self.scene.render_camera(w, h, cam_name)
                if self.cam_crop:
                    im = utils.realsense_resize_crop(im, im_type="rgb")
                obs[f"rgb_{cam_name}"] = im
                if self.segmentation_masks:
                    seg = self.scene.render_camera(w, h, cam_name, im_type="seg")
                    if self.cam_crop:
                        seg = utils.realsense_resize_crop(seg, im_type="seg")
                    obs[f"seg_{cam_name}"] = seg
        return obs

    def step(self, action):
        action_np = np.zeros(7)
        action_np[:3] = action.get("linear_velocity", np.zeros(3))

        action_np[3:6] = action.get("angular_velocity", np.zeros(3))
        action_np[6] = action.get("grip_open", 0)

        action_dict = dict()
        action_dict[self.gripper_name] = action_np

        self.scene.step(action_dict, self.workspace)
        obs = self.observe()
        success = self.is_success()
        reward = float(success)
        done = success
        info = dict(success=success)
        return obs, reward, done, info

    def oracle(self):
        return ScriptAgent(self, self.open_gripper_init)
