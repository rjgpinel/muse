import os
import argparse
from pathlib import Path
import pickle as pkl
import numpy as np
from PIL import Image

import gym
import muse.envs
from muse.core import constants
from muse.envs.utils import realsense_resize_crop
from mujoco_py.modder import TextureModder


def set_colors(scene, modder):
    modder.whiten_materials()
    for name in scene.sim.model.geom_names:
        if (
            "gripper" in name
            or "truss_arm" in name
            or "moment_arm" in name
            or "finger" in name
        ):
            modder.set_rgb(name, (255, 0, 0))
        if "table" in name:
            modder.set_rgb(name, (0, 255, 0))
        if "ft300" in name:
            modder.set_rgb(name, (0, 0, 255))


def get_args_parser():
    parser = argparse.ArgumentParser("Align sim and real", add_help=False)
    parser.add_argument("--path", type=str)
    return parser


def main(args):
    path = Path(args.path)

    env = gym.make(
        "Pick-v0",
        cam_resolution=constants.REALSENSE_RESOLUTION,
        cam_crop=False,
    )
    env.reset()
    scene = env.unwrapped.scene

    texture_modder = TextureModder(scene.sim)
    set_colors(scene, texture_modder)

    data = pkl.load(open(path, "rb"))
    gripper_pos = np.array(data["xyz_grip"])
    # cube_pos = np.array(data["xyz_cube"])
    # cube_pos = np.hstack(list(cube_pos) + [1, 0, 0, 0])

    scene.reset(
        mocap_target_pos=dict(left_gripper=gripper_pos),
        open_gripper=dict(left_gripper=True),
        workspace=env.unwrapped.workspace,
    )
    # scene.set_joint_qpos("cube0_joint", cube_pos)
    scene.warmup()

    color = (1, 0, 0)
    ims = []
    for cam_name in ["bravo", "charlie"]:
        sim_rgb = env.observe()[f"rgb_{cam_name}_camera"]
        im0 = Image.fromarray(sim_rgb)

        im1 = data[f"{cam_name}_img"]
        # im1 = realsense_resize_crop(im1)
        im1 = Image.fromarray(im1)

        res = Image.blend(im0, im1, 0.5)
        # res = np.vstack((im0, im1))
        ims.append(np.asarray(res))

    ims = np.hstack(ims)
    ims = Image.fromarray(ims)
    ims.show()
    ims.save("align.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Align sim and real", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
