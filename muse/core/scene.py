import numpy as np
import copy

from scipy.spatial.transform import Rotation

from mujoco_py import load_model_from_path
from mujoco_py import MjSim, MjViewer

from muse.core import constants, utils
from muse.core.modder import Modder
from muse.core.utils import (
    quat_to_euler,
    muj_to_std_quat,
    euler_to_quat,
    std_to_muj_quat,
    xmat_to_std_quat,
)


class Scene:
    def __init__(
        self,
        model_path="",
        viewer=False,
        domain_randomization=False,
        num_textures=None,
        textures_pack="bop",
        cam_xyz_noise=0.0,
        light_setup="default",
        arms_loaded=("left_arm",),
    ):
        if not model_path:
            model_path = utils.assets_dir() / "base_env.xml"
        else:
            model_path = utils.assets_dir() / model_path

        self.model = load_model_from_path(str(model_path))
        self.sim = MjSim(self.model)
        self._initial_state = copy.deepcopy(self.sim.get_state())
        self._arms_loaded = arms_loaded

        if viewer:
            self.viewer = MjViewer(self.sim)
            self._setup_viewer()
        else:
            self.viewer = None

        self.modder = Modder(
            self.sim,
            domain_randomization,
            num_textures=num_textures,
            textures_pack=textures_pack,
            cam_xyz_noise=cam_xyz_noise,
            light_setup=light_setup,
        )

        self.controller_dt = constants.CONTROLLER_DT
        self.max_tool_velocity = constants.MAX_TOOL_VELOCITY

        self._mocap_pos = dict()
        self._mocap_quat = dict()
        self._gripper_id = dict()
        self._init_arms()
        self._init_gripper_mocap()

    def reset(self, mocap_pos, mocap_quat, open_gripper, workspace):
        self.sim.set_state(self._initial_state)

        self.total_steps = 0

        self._init_arms()
        utils.reset_mocap_welds(self.sim)
        self._init_gripper_mocap()

        self._mocap_pos.update(mocap_pos)
        if mocap_quat is not None:
            self._mocap_quat.update(mocap_quat)

        # initialize grippers pose
        for name, target_pos in self._mocap_pos.items():
            target_quat = self._mocap_quat[name]
            self._set_mocap_target(name, target_pos, target_quat)

        self.sim.forward()

        # initialize grippers openness
        for name, gripper_target in open_gripper.items():
            self._init_gripper_state(name, gripper_target)

        self.modder.apply(workspace)

    def warmup(self):
        """
        warmup simulation by running simulation steps
        set mocap target and resolves penetrating objects

        WARNING: always call this function after setting all the model poses
        typically at the end of the reset of an environment before observe
        """
        for _ in range(constants.WARMUP_STEPS):
            self.sim.step()

    def step(self, action, workspace):
        for name_ee, action_ee in action.items():
            assert action_ee.shape[0] == 7
            # update the mocap pose and gripper target angles
            self._step_mocap(name_ee, action_ee[:3], action_ee[3:6], workspace)
            self._step_gripper(name_ee, action_ee[6:])
        # run simulation steps for the arms and gripper to match defined targets
        for i in range(constants.SIM_STEPS):
            self.sim.step()
            if self.viewer and self.total_steps % constants.RENDER_STEPS == 0:
                self.render()
            self.total_steps += 1

    def render(self):
        self.viewer.render()

    def render_camera(self, width, height, camera_name, im_type="rgb"):
        if im_type == "rgb":
            im = self.sim.render(width=width, height=height, camera_name=camera_name)
            im = np.flip(im, 0)
            return im
        elif im_type == "seg":
            seg = self.sim.render(
                width=width, height=height, camera_name=camera_name, segmentation=True
            )
            seg = np.flip(seg, 0)
            types = seg[:, :, 0]
            ids = seg[:, :, 1]
            return ids
        else:
            raise ValueError(f"Unknow image type: {im_type}")

    def get_site_pos(self, name):
        return np.copy(self.sim.data.get_site_xpos(name))

    def get_site_quat(self, name):
        return np.copy(xmat_to_std_quat(self.sim.data.get_site_xmat(name)))

    def get_geom_pos(self, name):
        return np.copy(self.sim.data.get_geom_xpos(name))

    def get_geom_quat(self, name):
        return np.copy(
            std_to_muj_quat(xmat_to_std_quat(self.sim.data.get_geom_xmat(name)))
        )

    def get_body_pos(self, name):
        return np.copy(self.sim.data.get_body_xpos(name))

    def get_body_quat(self, name):
        return np.copy(std_to_muj_quat(self.sim.data.get_body_xquat(name)))

    def get_joint_qpos(self, name):
        return np.copy(self.sim.data.get_joint_qpos(name))

    def get_ee_pos(self, name):
        return np.copy(self.sim.data.get_mocap_pos(f"{name}_mocap"))

    def get_ee_quat(self, name):
        return np.copy(self.sim.data.get_mocap_quat(f"{name}_mocap"))

    def get_gripper_qpos(self, name):
        return self.get_joint_qpos(f"{name}_finger_1_joint")

    def get_geom_size(self, name):
        geom_id = self.model.geom_name2id(name)
        geom_size = self.model.geom_size[geom_id]
        return geom_size

    def set_joint_qpos(self, name, qpos):
        self.sim.data.set_joint_qpos(name, qpos)
        qvel = self.sim.data.get_joint_qvel(name)
        self.sim.data.set_joint_qvel(name, np.zeros_like(qvel))
        self.sim.forward()

    def _step_mocap(self, name, linvel, angvel, workspace):
        linvel = linvel * self.controller_dt
        linvel = np.clip(
            linvel, -constants.MAX_TOOL_VELOCITY[0], constants.MAX_TOOL_VELOCITY[0]
        )
        target_pos = self._mocap_pos[name] + linvel
        target_pos = np.clip(target_pos, workspace[0], workspace[1])

        angvel = angvel * self.controller_dt
        angvel = np.clip(
            angvel, -constants.MAX_TOOL_VELOCITY[1], constants.MAX_TOOL_VELOCITY[1]
        )
        mocap_quat = muj_to_std_quat(self._mocap_quat[name])
        mocap_orn = quat_to_euler(mocap_quat, False)
        target_orn = mocap_orn + angvel
        # target_orn = np.clip(
        # target_orn,
        # [-np.pi, -np.pi, -np.pi],
        # [np.pi, np.pi, np.pi],
        # )
        target_quat = euler_to_quat(target_orn, False)
        target_quat = std_to_muj_quat(target_quat)
        self._set_mocap_target(name, target_pos, target_quat)

    def _step_gripper(self, name, openness):
        gripper_id = self._gripper_id[name]
        openness = np.clip(openness, -1, 1)
        action = constants.GRIPPER_ACTION_SCALING * np.clip(-openness, -1, 1)
        idx = self.sim.model.jnt_qposadr[self.sim.model.actuator_trnid[gripper_id, 0]]
        self.sim.data.ctrl[gripper_id] = self.sim.data.qpos[idx] + action

    def _set_mocap_target(self, name, target_pos=None, target_quat=None):
        if target_pos is not None:
            self._mocap_pos[name] = target_pos
        if target_quat is not None:
            self._mocap_quat[name] = target_quat
        self.sim.data.set_mocap_pos(f"{name}_mocap", self._mocap_pos[name])
        self.sim.data.set_mocap_quat(f"{name}_mocap", self._mocap_quat[name])
        self.sim.forward()

    def _setup_viewer(self):
        body_id = self.sim.model.body_name2id("left_gripper_mocap")
        lookat = self.sim.data.body_xpos[body_id]
        viewer_cam = self.viewer.cam
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.6
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = -20.0

    def _init_gripper_mocap(self):
        for arm_name in self._arms_loaded:
            if arm_name == "left_arm":
                gripper_name = "left_gripper"
                gripper_pos = [-0.45, 0, 0.2]
                gripper_quat = utils.euler_to_quat((0, 180, -90), degrees=True)
                gripper_quat = utils.std_to_muj_quat(gripper_quat)
                gripper_id = 0
            elif arm_name == "right_arm":
                gripper_name = "right_gripper"
                gripper_pos = [0.45, 0, 0.2]
                gripper_quat = utils.euler_to_quat((-76.5, 180, 90), degrees=True)
                gripper_quat = utils.std_to_muj_quat(gripper_quat)
                gripper_id = 1
            self._mocap_pos[gripper_name] = gripper_pos
            self._mocap_quat[gripper_name] = gripper_quat
            self._gripper_id[gripper_name] = gripper_id

    def _init_arms(self):
        initial_q_pos = dict()
        for arm_name in self._arms_loaded:
            if arm_name == "left_arm":
                arm_q_pos = dict(
                    left_shoulder_pan_joint=-0.9773843811168246,
                    left_shoulder_lift_joint=-1.7627825445142729,
                    left_elbow_joint=-2.321287905152458,
                    left_wrist_1_joint=-1.1344640137963142,
                    left_wrist_2_joint=-2.199114857512855,
                    left_wrist_3_joint=-2.3387411976724017,
                )
            elif arm_name == "right_arm":
                arm_q_pos = dict(
                    right_shoulder_pan_joint=-0.575959,
                    right_shoulder_lift_joint=-2.687807,
                    right_elbow_joint=-0.767945,
                    right_wrist_1_joint=0.314159,
                    right_wrist_2_joint=-1.047198,
                    right_wrist_3_joint=0,
                )
            initial_q_pos.update(arm_q_pos)

        for name, val in initial_q_pos.items():
            self.sim.data.set_joint_qpos(name, val)
            self.sim.data.set_joint_qvel(name, np.zeros_like(val))
        self.sim.forward()

    def _init_gripper_state(self, gripper_name, open_gripper=True):
        gripper_id = self._gripper_id[gripper_name]
        q_open, q_close = self.sim.model.actuator_ctrlrange[gripper_id]
        if open_gripper:
            q = q_open
        else:
            q = q_close

        initial_q_pos = dict(
            finger_1_joint=q,
            finger_1_truss_arm_joint=q,
            finger_1_safety_shield_joint=q,
            finger_1_tip_joint=q,
            finger_2_joint=q,
            finger_2_truss_arm_joint=q,
            finger_2_safety_shield_joint=q,
            finger_2_tip_joint=q,
        )

        if gripper_name == "right_gripper":
            initial_q_pos["finger_1_safety_shield_joint"] = q
            initial_q_pos["finger_2_safety_shield_joint"] = q

        for finger_name, val in initial_q_pos.items():
            name = f"{gripper_name}_{finger_name}"
            self.sim.data.set_joint_qpos(name, val)
            self.sim.data.set_joint_qvel(name, np.zeros_like(val))
            self.sim.data.ctrl[gripper_id] = val
        self.sim.forward()
