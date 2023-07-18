import copy
from mujoco_py.modder import TextureModder, LightModder, CameraModder, BaseModder
import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb
from pathlib import Path

from PIL import Image
from muse.core import utils
from muse.core import constants


class Modder:
    def __init__(
        self,
        sim,
        domain_randomization,
        num_textures=None,
        modder_seed=0,
        textures_pack="bop",
        cam_xyz_noise=0.0,
        light_setup="default",
        # light_rand=False,
        light_rand=True,
        # color_rand=False,
        color_rand=True,
        # cam_rand=False,
        cam_rand=True,
        bop_prob=1.0,
        # bop_prob=0.0,
    ):
        # self._np_random = np.random.RandomState(modder_seed)
        self._np_random = np.random
        self.sim = sim
        self.texture_modder = TextureModder(self.sim)
        self.camera_modder = CameraModder(self.sim)
        self.light_modder = LightModder(self.sim)

        self.light_rand = light_rand
        self.color_rand = color_rand
        self.cam_rand = cam_rand
        self.bop_prob = bop_prob

        # reference infos store obj infos during first call
        # then use it as reference during next resets
        # noise adds up across reset otherwize
        self.camera_infos = dict()
        self.geom_infos = dict()
        self.num_textures = num_textures
        self.domain_randomization = domain_randomization
        self.cam_xyz_noise = cam_xyz_noise
        self.light_setup = light_setup

        self.textures_paths = [
            utils.assets_dir() / constants.TEXTURES_PATH[textures_pack],
        ]

        self.textures = []
        for textures_path in self.textures_paths:
            for ext in ["*.jpg", "*.png"]:
                self.textures.extend(textures_path.glob(ext))

        print(f"{len(self.textures)} textures loaded.")

        self._np_random.shuffle(self.textures)
        if num_textures is not None:
            self.textures = self.textures[:num_textures]
        if self.domain_randomization and not self.textures:
            print("Warning. BOP Textures are not installed.")

    def rand_hsv(self, name, color=None):
        if color is None:
            if name not in self.geom_infos:
                geom_id = self.sim.model.geom_name2id(name)
                rgba = self.sim.model.geom_rgba[geom_id]
                self.geom_infos[name] = rgba
            else:
                rgba = self.geom_infos[name]
            r, g, b = rgba[:3]
        else:
            r, g, b = color

        if self.color_rand:
            h, s, v = rgb_to_hsv(r, g, b)
            # h += np.random.uniform(low=-0.1, high=0.1)
            # Fixed implementation
            h += self._np_random.uniform(low=-0.05, high=0.05)
            if h < 0:
                h = 1 + h

            # Fixed implementation
            s += self._np_random.uniform(low=-0.1, high=0.1)
            v += self._np_random.uniform(low=-0.1, high=0.1)

            # Old implementation
            # s, v = np.random.uniform(low=0.5, high=1, size=(2,))

            h = np.clip(h, 0, 1)
            s = np.clip(s, 0, 1)
            v = np.clip(v, 0, 1)

            r, g, b = hsv_to_rgb(h, s, v)
        rgb = np.array((255 * r, 255 * g, 255 * b), dtype=np.uint8)
        self.texture_modder.set_rgb(name, rgb)

    def rand_color(self, name):
        rgb = self._np_random.uniform(0, 255, 3, dtype=np.uint8)
        self.texture_modder.set_rgb(name, rgb)

    def set_alpha(self, name, alpha):
        geom_id = self.sim.model.geom_name2id(name)
        self.sim.model.geom_rgba[geom_id][-1] = alpha

    def rand_alpha(self, name):
        prob = self._np_random.rand()
        if prob <= 0.5:
            self.set_alpha(name, 1)
        else:
            self.set_alpha(name, 0)

    def set_bop_texture(self, name):
        texture_path = self._np_random.choice(self.textures)
        new_bitmap = np.asarray(Image.open(str(texture_path)))
        new_bitmap_w, new_bitmap_h, _ = new_bitmap.shape
        bitmap = self.texture_modder.get_texture(name).bitmap
        ori_bitmap_w, ori_bitmap_h, _ = bitmap.shape

        bitmap_w = min(ori_bitmap_w, new_bitmap_w)
        bitmap_h = min(ori_bitmap_h, new_bitmap_h)

        for i in range(int(np.ceil(ori_bitmap_w / bitmap_w))):
            for j in range(int(np.ceil(ori_bitmap_h / bitmap_h))):
                bitmap_part_w, bitmap_part_h, _ = bitmap[
                    i * bitmap_w : (i + 1) * bitmap_w,
                    j * bitmap_h : (j + 1) * bitmap_h,
                    :,
                ].shape
                bitmap[
                    i * bitmap_w : (i + 1) * bitmap_w,
                    j * bitmap_h : (j + 1) * bitmap_h,
                    :,
                ] = new_bitmap[:bitmap_part_w, :bitmap_part_h, :]
        self.texture_modder.upload_texture(name)

    def set_textures(self):
        # important to reset all material colors before applying textures
        self.texture_modder.whiten_materials()

        # TODO: Put obj keyword on XML templates - Objects won't have texture augmentation
        obj_names = ["cube", "nut", "screw", "marker", "pile", "obstacle"]

        for name in list(self.sim.model.geom_names):
            is_obj = False
            for obj_name in obj_names:
                if obj_name in name:
                    is_obj = True
            if is_obj:
                self.rand_hsv(name)
            else:
                apply_bop = self._np_random.rand() < self.bop_prob
                if self.textures and apply_bop:
                    self.set_bop_texture(name)
                else:
                    self.texture_modder.rand_all(name)

    def set_light(self, light_name):
        light_modder = self.light_modder

        active = True
        if self.domain_randomization and self.light_rand:
            ambient = self._np_random.uniform(0.0, 0.3, size=(3,))
            diffuse = self._np_random.uniform(0.0, 0.6, size=(3,))
            specular = self._np_random.uniform(0.0, 0.3, size=(3,))

            # sample in a portion of sphere around the table
            # theta = [0, pi/2, pi] -> x = [1, 0, -1], y = [0, 1, 0]
            # phi = [0, pi/2] -> z = [0, 1]
            theta = self._np_random.uniform(0, np.pi)
            phi = self._np_random.uniform(1 / 5 * np.pi / 2, 4 / 5 * np.pi / 2)
            dist = self._np_random.uniform(1.0, 3.0)

            cast_shadow = self._np_random.randint(2)
        else:
            if self.light_setup == "default":
                ambient = np.full(3, 0.3)
                diffuse = np.full(3, 0.6)
                specular = np.full(3, 0.3)

                theta = np.pi / 2
                phi = np.pi / 4
                dist = 2.0

                cast_shadow = False

            elif self.light_setup == "low_light":
                ambient = np.full(3, 0.1)
                diffuse = np.full(3, 0.1)
                specular = np.full(3, 0.1)

                theta = np.pi / 2
                phi = np.pi / 8
                dist = 2.0

                cast_shadow = True

            elif self.light_setup == "rgb_light":
                rgb_idx = self._np_random.randint(3)

                ambient = np.array([0.0, 0.0, 0.0])
                diffuse = np.array([0.0, 0.0, 0.0])
                specular = np.array([0.0, 0.0, 0.0])

                ambient[rgb_idx] = 0.4
                diffuse[rgb_idx] = 0.6
                specular[rgb_idx] = 0.4

                theta = np.pi / 2
                phi = np.pi / 4
                dist = 2.0

                cast_shadow = False

        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)

        pos = dist * np.array([x, y, z])
        direction = -pos

        light_modder.set_pos(light_name, pos)
        light_modder.set_dir(light_name, direction)
        light_modder.set_ambient(light_name, ambient)
        light_modder.set_diffuse(light_name, diffuse)
        light_modder.set_specular(light_name, specular)
        light_modder.set_castshadow(light_name, cast_shadow)
        light_modder.set_active(light_name, active)

    def set_camera(self, cam_name, workspace):
        workspace_center = workspace.mean(0)
        workspace_center[2] = 0.0

        delta_pos = self._np_random.uniform(
            low=-self.cam_xyz_noise, high=self.cam_xyz_noise, size=3
        )

        cam_modder = self.camera_modder
        if cam_name not in self.camera_infos:
            cam_pos = cam_modder.get_pos(cam_name)
            cam_quat = cam_modder.get_quat(cam_name)
            cam_quat = utils.muj_to_std_quat(cam_quat)
            cam_euler = utils.quat_to_euler(cam_quat, degrees=False)
            self.camera_infos[cam_name] = dict(
                pos=cam_pos.copy(),
                euler=cam_euler.copy(),
                fovy=constants.REALSENSE_FOV,
            )

        cam_pos = self.camera_infos[cam_name]["pos"]
        cam_euler = self.camera_infos[cam_name]["euler"]
        cam_fov = self.camera_infos[cam_name]["fovy"]

        if (
            self.domain_randomization
            and cam_name != "left_mounted_camera"
            and self.cam_rand
        ):
            # distance to center of workspace +/- 10cm
            delta_distance = self._np_random.uniform(low=-0.1, high=0.1, size=1)
            cam_pos = workspace_center + (1 + delta_distance) * (
                cam_pos - workspace_center
            )

            # euler angles +/- 0.05 rad = 2.87 deg
            delta_euler = self._np_random.uniform(low=-0.05, high=0.05, size=1)
            cam_euler = cam_euler * (1 + delta_euler)

            cam_fov += self._np_random.uniform(low=-1, high=1, size=1)

        cam_pos += delta_pos
        cam_modder.set_pos(cam_name, cam_pos)

        cam_quat = utils.euler_to_quat(cam_euler, degrees=False)
        cam_quat = utils.std_to_muj_quat(cam_quat)
        cam_modder.set_quat(cam_name, cam_quat)

        cam_modder.set_fovy(cam_name, cam_fov)

    def apply(self, workspace):
        for cam_name in self.sim.model.camera_names:
            self.set_camera(cam_name, workspace)

        if self.domain_randomization:
            self.set_textures()

        for light_name in self.sim.model.light_names:
            self.set_light(light_name)


class DynamicsModder(BaseModder):
    """
    Modder for various dynamics properties of the mujoco model, such as friction, damping, etc.
    This can be used to modify parameters stored in MjModel (ie friction, damping, etc.) as
    well as optimizer parameters stored in PyMjOption (i.e.: medium density, viscosity, etc.)
    To modify a parameter, use the parameter to be changed as a keyword argument to
    self.mod and the new value as the value for that argument. Supports arbitrary many
    modifications in a single step. Example use:
        sim = MjSim(...)
        modder = DynamicsModder(sim)
        modder.mod("element1_name", "attr1", new_value1)
        modder.mod("element2_name", "attr2", new_value2)
        ...
        modder.update()

    NOTE: It is necessary to perform modder.update() after performing all modifications to make sure
        the changes are propagated

    NOTE: A full list of supported randomizable parameters can be seen by calling modder.dynamics_parameters

    NOTE: When modifying parameters belonging to MjModel.opt (e.g.: density, viscosity), no name should
        be specified (set it as None in mod(...)). This is because opt does not have a name attribute
        associated with it

    Args:
        sim (MjSim): Mujoco sim instance

        random_state (RandomState): instance of np.random.RandomState

        randomize_density (bool): If True, randomizes global medium density

        randomize_viscosity (bool): If True, randomizes global medium viscosity

        density_perturbation_ratio (float): Relative (fraction) magnitude of default density randomization

        viscosity_perturbation_ratio:  Relative (fraction) magnitude of default viscosity randomization

        body_names (None or list of str): list of bodies to use for randomization. If not provided, all
            bodies in the model are randomized.

        randomize_position (bool): If True, randomizes body positions

        randomize_quaternion (bool): If True, randomizes body quaternions

        randomize_inertia (bool): If True, randomizes body inertias (only applicable for non-zero mass bodies)

        randomize_mass (bool): If True, randomizes body masses (only applicable for non-zero mass bodies)

        position_perturbation_size (float): Magnitude of body position randomization

        quaternion_perturbation_size (float): Magnitude of body quaternion randomization (angle in radians)

        inertia_perturbation_ratio (float): Relative (fraction) magnitude of body inertia randomization

        mass_perturbation_ratio (float): Relative (fraction) magnitude of body mass randomization

        geom_names (None or list of str): list of geoms to use for randomization. If not provided, all
            geoms in the model are randomized.

        randomize_friction (bool): If True, randomizes geom frictions

        randomize_solref (bool): If True, randomizes geom solrefs

        randomize_solimp (bool): If True, randomizes geom solimps

        friction_perturbation_ratio (float): Relative (fraction) magnitude of geom friction randomization

        solref_perturbation_ratio (float): Relative (fraction) magnitude of geom solref randomization

        solimp_perturbation_ratio (float): Relative (fraction) magnitude of geom solimp randomization

        joint_names (None or list of str): list of joints to use for randomization. If not provided, all
            joints in the model are randomized.

        randomize_stiffness (bool): If True, randomizes joint stiffnesses

        randomize_frictionloss (bool): If True, randomizes joint frictionlosses

        randomize_damping (bool): If True, randomizes joint dampings

        randomize_armature (bool): If True, randomizes joint armatures

        stiffness_perturbation_ratio (float): Relative (fraction) magnitude of joint stiffness randomization

        frictionloss_perturbation_size (float): Magnitude of joint frictionloss randomization

        damping_perturbation_size (float): Magnitude of joint damping randomization

        armature_perturbation_size (float): Magnitude of joint armature randomization
    """

    def __init__(
        self,
        sim,
        random_state=None,
        # Opt parameters
        randomize_density=True,
        randomize_viscosity=True,
        density_perturbation_ratio=0.1,
        viscosity_perturbation_ratio=0.1,
        # Body parameters
        body_names=None,
        randomize_position=True,
        randomize_quaternion=True,
        randomize_inertia=True,
        randomize_mass=True,
        position_perturbation_size=0.02,
        quaternion_perturbation_size=0.02,
        inertia_perturbation_ratio=0.02,
        mass_perturbation_ratio=0.02,
        # Geom parameters
        geom_names=None,
        randomize_friction=True,
        randomize_solref=True,
        randomize_solimp=True,
        friction_perturbation_ratio=0.1,
        solref_perturbation_ratio=0.1,
        solimp_perturbation_ratio=0.1,
        # Joint parameters
        joint_names=None,
        randomize_stiffness=True,
        randomize_frictionloss=True,
        randomize_damping=True,
        randomize_armature=True,
        stiffness_perturbation_ratio=0.1,
        frictionloss_perturbation_size=0.05,
        damping_perturbation_size=0.01,
        armature_perturbation_size=0.01,
    ):
        super().__init__(sim=sim, random_state=random_state)

        # Setup relevant values
        self.dummy_bodies = set()
        # Find all bodies that don't have any mass associated with them
        for body_name in self.sim.model.body_names:
            body_id = self.sim.model.body_name2id(body_name)
            if self.sim.model.body_mass[body_id] == 0:
                self.dummy_bodies.add(body_name)

        # Get all values to randomize
        self.body_names = (
            list(self.sim.model.body_names) if body_names is None else body_names
        )
        self.geom_names = (
            list(self.sim.model.geom_names) if geom_names is None else geom_names
        )
        self.joint_names = (
            list(self.sim.model.joint_names) if joint_names is None else joint_names
        )

        # Setup randomization settings
        # Each dynamics randomization group has its set of randomizable parameters, each of which has
        # its own settings ["randomize": whether its actively being randomized, "perturbation": the (potentially)
        # relative magnitude of the randomization to use, "type": either "ratio" or "size" (relative or absolute
        # perturbations), and "clip": (low, high) values to clip the final perturbed value by]
        self.opt_randomizations = {
            "density": {
                "randomize": randomize_density,
                "perturbation": density_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
            "viscosity": {
                "randomize": randomize_viscosity,
                "perturbation": viscosity_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
        }

        self.body_randomizations = {
            "position": {
                "randomize": randomize_position,
                "perturbation": position_perturbation_size,
                "type": "size",
                "clip": (-np.inf, np.inf),
            },
            "quaternion": {
                "randomize": randomize_quaternion,
                "perturbation": quaternion_perturbation_size,
                "type": "size",
                "clip": (-np.inf, np.inf),
            },
            "inertia": {
                "randomize": randomize_inertia,
                "perturbation": inertia_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
            "mass": {
                "randomize": randomize_mass,
                "perturbation": mass_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
        }

        self.geom_randomizations = {
            "friction": {
                "randomize": randomize_friction,
                "perturbation": friction_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
            "solref": {
                "randomize": randomize_solref,
                "perturbation": solref_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, 1.0),
            },
            "solimp": {
                "randomize": randomize_solimp,
                "perturbation": solimp_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
        }

        self.joint_randomizations = {
            "stiffness": {
                "randomize": randomize_stiffness,
                "perturbation": stiffness_perturbation_ratio,
                "type": "ratio",
                "clip": (0.0, np.inf),
            },
            "frictionloss": {
                "randomize": randomize_frictionloss,
                "perturbation": frictionloss_perturbation_size,
                "type": "size",
                "clip": (0.0, np.inf),
            },
            "damping": {
                "randomize": randomize_damping,
                "perturbation": damping_perturbation_size,
                "type": "size",
                "clip": (0.0, np.inf),
            },
            "armature": {
                "randomize": randomize_armature,
                "perturbation": armature_perturbation_size,
                "type": "size",
                "clip": (0.0, np.inf),
            },
        }

        # Store defaults so we don't loss track of the original (non-perturbed) values
        self.opt_defaults = None
        self.body_defaults = None
        self.geom_defaults = None
        self.joint_defaults = None
        self.save_defaults()

    def save_defaults(self):
        """
        Grabs the current values for all parameters in sim and stores them as default values
        """
        self.opt_defaults = {
            None: {  # no name associated with the opt parameters
                "density": self.sim.model.opt.density,
                "viscosity": self.sim.model.opt.viscosity,
            }
        }

        self.body_defaults = {}
        for body_name in self.sim.model.body_names:
            body_id = self.sim.model.body_name2id(body_name)
            self.body_defaults[body_name] = {
                "position": np.array(self.sim.model.body_pos[body_id]),
                "quaternion": np.array(self.sim.model.body_quat[body_id]),
                "inertia": np.array(self.sim.model.body_inertia[body_id]),
                "mass": self.sim.model.body_mass[body_id],
            }

        self.geom_defaults = {}
        for geom_name in self.sim.model.geom_names:
            geom_id = self.sim.model.geom_name2id(geom_name)
            self.geom_defaults[geom_name] = {
                "friction": np.array(self.sim.model.geom_friction[geom_id]),
                "solref": np.array(self.sim.model.geom_solref[geom_id]),
                "solimp": np.array(self.sim.model.geom_solimp[geom_id]),
            }

        self.joint_defaults = {}
        for joint_name in self.sim.model.joint_names:
            joint_id = self.sim.model.joint_name2id(joint_name)
            dof_idx = [
                i for i, v in enumerate(self.sim.model.dof_jntid) if v == joint_id
            ]
            self.joint_defaults[joint_name] = {
                "stiffness": self.sim.model.jnt_stiffness[joint_id],
                "frictionloss": np.array(self.sim.model.dof_frictionloss[dof_idx]),
                "damping": np.array(self.sim.model.dof_damping[dof_idx]),
                "armature": np.array(self.sim.model.dof_armature[dof_idx]),
            }

    def restore_defaults(self):
        """
        Restores the default values curently saved in this modder
        """
        # Loop through all defaults and set the default value in sim
        for group_defaults in (
            self.opt_defaults,
            self.body_defaults,
            self.geom_defaults,
            self.joint_defaults,
        ):
            for name, defaults in group_defaults.items():
                for attr, default_val in defaults.items():
                    self.mod(name=name, attr=attr, val=default_val)

        # Make sure changes propagate in sim
        self.update()

    def randomize(self):
        """
        Randomizes all enabled dynamics parameters in the simulation
        """
        for group_defaults, group_randomizations, group_randomize_names in zip(
            (
                self.opt_defaults,
                self.body_defaults,
                self.geom_defaults,
                self.joint_defaults,
            ),
            (
                self.opt_randomizations,
                self.body_randomizations,
                self.geom_randomizations,
                self.joint_randomizations,
            ),
            ([None], self.body_names, self.geom_names, self.joint_names),
        ):
            for name in group_randomize_names:
                # Randomize all parameters associated with this element
                for attr, default_val in group_defaults[name].items():
                    val = copy.copy(default_val)
                    settings = group_randomizations[attr]
                    if settings["randomize"]:
                        # Randomize accordingly, and clip the final perturbed value
                        perturbation = (
                            np.random.rand()
                            if type(val) in {int, float}
                            else np.random.rand(*val.shape)
                        )
                        perturbation = settings["perturbation"] * (
                            -1 + 2 * perturbation
                        )
                        val = (
                            val + perturbation
                            if settings["type"] == "size"
                            else val * (1.0 + perturbation)
                        )

                        val = np.clip(val, *settings["clip"])
                    # Modify this value
                    self.mod(name=name, attr=attr, val=val)

        # Make sure changes propagate in sim
        self.update()

    def update_sim(self, sim):
        """
        In addition to super method, update internal default values to match the current values from
        (the presumably new) @sim.

        Args:
            sim (MjSim): MjSim object
        """
        self.sim = sim
        self.save_defaults()

    def update(self):
        """
        Propagates the changes made up to this point through the simulation
        """
        self.sim.forward()

    def mod(self, name, attr, val):
        """
        General method to modify dynamics parameter @attr to be new value @val, associated with element @name.

        Args:
            name (str): Name of element to modify parameter. This can be a body, geom, or joint name. If modifying
                an opt parameter, this should be set to None
            attr (str): Name of the dynamics parameter to modify. Valid options are self.dynamics_parameters
            val (int or float or n-array): New value(s) to set for the given dynamics parameter. The type of this
                argument should match the expected type for the given parameter.
        """
        # Make sure specified parameter is valid, and then modify it
        assert attr in self.dynamics_parameters, (
            "Invalid dynamics parameter specified! Supported parameters are: {};"
            " requested: {}".format(self.dynamics_parameters, attr)
        )
        # Modify the requested parameter (uses a clean way to programmatically call the appropriate method)
        getattr(self, f"mod_{attr}")(name, val)

    def mod_density(self, name=None, val=0.0):
        """
        Modifies the global medium density of the simulation.
        See http://www.mujoco.org/book/XMLreference.html#option for more details.

        Args:
            name (str): Name for this element. Should be left as None (opt has no name attribute)
            val (float): New density value.
        """
        # Make sure inputs are of correct form
        assert name is None, "No name should be specified if modding density!"

        # Modify this value
        self.sim.model.opt.density = val

    def mod_viscosity(self, name=None, val=0.0):
        """
        Modifies the global medium viscosity of the simulation.
        See http://www.mujoco.org/book/XMLreference.html#option for more details.

        Args:
            name (str): Name for this element. Should be left as None (opt has no name attribute)
            val (float): New viscosity value.
        """
        # Make sure inputs are of correct form
        assert name is None, "No name should be specified if modding density!"

        # Modify this value
        self.sim.model.opt.viscosity = val

    def mod_position(self, name, val=(0, 0, 0)):
        """
        Modifies the @name's relative body position within the simulation.
        See http://www.mujoco.org/book/XMLreference.html#body for more details.

        Args:
            name (str): Name for this element.
            val (3-array): New (x, y, z) relative position.
        """
        # Modify this value
        body_id = self.sim.model.body_name2id(name)
        self.sim.model.body_pos[body_id] = np.array(val)

    def mod_quaternion(self, name, val=(1, 0, 0, 0)):
        """
        Modifies the @name's relative body orientation (quaternion) within the simulation.
        See http://www.mujoco.org/book/XMLreference.html#body for more details.

        Note: This method automatically normalizes the inputted value.

        Args:
            name (str): Name for this element.
            val (4-array): New (w, x, y, z) relative quaternion.
        """
        # Normalize the inputted value
        val = np.array(val) / np.linalg.norm(val)
        # Modify this value
        body_id = self.sim.model.body_name2id(name)
        self.sim.model.body_quat[body_id] = val

    def mod_inertia(self, name, val):
        """
        Modifies the @name's relative body inertia within the simulation.
        See http://www.mujoco.org/book/XMLreference.html#body for more details.

        Args:
            name (str): Name for this element.
            val (3-array): New (ixx, iyy, izz) diagonal values in the inertia matrix.
        """
        # Modify this value if it's not a dummy body
        if name not in self.dummy_bodies:
            body_id = self.sim.model.body_name2id(name)
            self.sim.model.body_inertia[body_id] = np.array(val)

    def mod_mass(self, name, val):
        """
        Modifies the @name's mass within the simulation.
        See http://www.mujoco.org/book/XMLreference.html#body for more details.

        Args:
            name (str): Name for this element.
            val (float): New mass.
        """
        # Modify this value if it's not a dummy body
        if name not in self.dummy_bodies:
            body_id = self.sim.model.body_name2id(name)
            self.sim.model.body_mass[body_id] = val

    def mod_friction(self, name, val):
        """
        Modifies the @name's geom friction within the simulation.
        See http://www.mujoco.org/book/XMLreference.html#geom for more details.

        Args:
            name (str): Name for this element.
            val (3-array): New (sliding, torsional, rolling) friction values.
        """
        # Modify this value
        geom_id = self.sim.model.geom_name2id(name)
        self.sim.model.geom_friction[geom_id] = np.array(val)

    def mod_solref(self, name, val):
        """
        Modifies the @name's geom contact solver parameters within the simulation.
        See http://www.mujoco.org/book/modeling.html#CSolver for more details.

        Args:
            name (str): Name for this element.
            val (2-array): New (timeconst, dampratio) solref values.
        """
        # Modify this value
        geom_id = self.sim.model.geom_name2id(name)
        self.sim.model.geom_solref[geom_id] = np.array(val)

    def mod_solimp(self, name, val):
        """
        Modifies the @name's geom contact solver impedance parameters within the simulation.
        See http://www.mujoco.org/book/modeling.html#CSolver for more details.

        Args:
            name (str): Name for this element.
            val (5-array): New (dmin, dmax, width, midpoint, power) solimp values.
        """
        # Modify this value
        geom_id = self.sim.model.geom_name2id(name)
        self.sim.model.geom_solimp[geom_id] = np.array(val)

    def mod_stiffness(self, name, val):
        """
        Modifies the @name's joint stiffness within the simulation.
        See http://www.mujoco.org/book/XMLreference.html#joint for more details.

        NOTE: If the stiffness is already at 0, we IGNORE this value since a non-stiff joint (i.e.: free-turning)
            joint is fundamentally different than a stiffened joint)

        Args:
            name (str): Name for this element.
            val (float): New stiffness.
        """
        # Modify this value (only if there is stiffness to begin with)
        jnt_id = self.sim.model.joint_name2id(name)
        if self.sim.model.jnt_stiffness[jnt_id] != 0:
            self.sim.model.jnt_stiffness[jnt_id] = val

    def mod_frictionloss(self, name, val):
        """
        Modifies the @name's joint frictionloss within the simulation.
        See http://www.mujoco.org/book/XMLreference.html#joint for more details.

        NOTE: If the requested joint is a free joint, it will be ignored since it does not
            make physical sense to have friction loss associated with this joint (air drag / damping
            is already captured implicitly by the medium density / viscosity values)

        Args:
            name (str): Name for this element.
            val (float): New friction loss.
        """
        # Modify this value (only if it's not a free joint)
        jnt_id = self.sim.model.joint_name2id(name)
        # FIXME: make it back to normal
        # if self.sim.model.jnt_type[jnt_id] != 0:
        dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == jnt_id]
        self.sim.model.dof_frictionloss[dof_idx] = val

    def mod_damping(self, name, val):
        """
        Modifies the @name's joint damping within the simulation.
        See http://www.mujoco.org/book/XMLreference.html#joint for more details.

        NOTE: If the requested joint is a free joint, it will be ignored since it does not
            make physical sense to have damping associated with this joint (air drag / damping
            is already captured implicitly by the medium density / viscosity values)

        Args:
            name (str): Name for this element.
            val (float): New damping.
        """
        # Modify this value (only if it's not a free joint)
        jnt_id = self.sim.model.joint_name2id(name)
        if self.sim.model.jnt_type[jnt_id] != 0:
            dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == jnt_id]
            self.sim.model.dof_damping[dof_idx] = val

    def mod_armature(self, name, val):
        """
        Modifies the @name's joint armature within the simulation.
        See http://www.mujoco.org/book/XMLreference.html#joint for more details.

        Args:
            name (str): Name for this element.
            val (float): New armature.
        """
        # Modify this value (only if it's not a free joint)
        jnt_id = self.sim.model.joint_name2id(name)
        if self.sim.model.jnt_type[jnt_id] != 0:
            dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == jnt_id]
            self.sim.model.dof_armature[dof_idx] = val

    @property
    def dynamics_parameters(self):
        """
        Returns:
            set: All dynamics parameters that can be randomized using this modder.
        """
        return {
            # Opt parameters
            "density",
            "viscosity",
            # Body parameters
            "position",
            "quaternion",
            "inertia",
            "mass",
            # Geom parameters
            "friction",
            "solref",
            "solimp",
            # Joint parameters
            "stiffness",
            "frictionloss",
            "damping",
            "armature",
        }

    @property
    def opt(self):
        """
        Returns:
             PyMjOption: MjModel sim options
        """
        return self.sim.model.opt
