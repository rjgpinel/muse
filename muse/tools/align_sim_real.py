import pickle as pkl
from pathlib import Path
import os
import numpy as np
import argparse
import skvideo.io

from PIL import Image

import muse.envs
import gym


def get_args_parser():
    parser = argparse.ArgumentParser("Align sim and real", add_help=False)
    parser.add_argument("--dataset-path", default="", type=str)
    parser.add_argument("--seed", default=5000, type=int)
    parser.add_argument("--step", default=0, type=int)
    parser.add_argument("--cam", default="charlie_camera", type=str)
    parser.add_argument("--split", dest="blend", action="store_false")
    parser.add_argument("--video", dest="video", action="store_true")
    parser.set_defaults(blend=True, video=False)
    return parser


def load_data(filename, filename_infos, cam_name):
    with open(filename, "rb") as f:
        obs, action = pkl.load(f)
    gripper_pos, gripper_orn = obs["gripper_pose"]
    real_rgb = obs[cam_name]

    with open(filename_infos, "rb") as f:
        data = pkl.load(f)
        cube_pos = data["cubes_position"][-1]

    cube_pos = np.hstack(list(cube_pos) + [1, 0, 0, 0])

    return real_rgb, gripper_pos, cube_pos


def generate_sim_im(env, gripper_pos, cube_pos, cam_name):
    env.unwrapped.scene.reset(
        mocap_target_pos=gripper_pos,
        open_gripper=True,
        workspace=env.unwrapped.workspace,
    )
    env.unwrapped.scene.set_joint_qpos("cube0_joint", cube_pos)
    sim_rgb = env.observe()[cam_name]
    return sim_rgb


def main(args):
    path = Path(args.dataset_path)
    low_res = (224, 224)
    high_res = (800, 800)
    env = gym.make("Pick-v0", viewer=False, cam_render=True)
    env.reset()

    cam_name = f"rgb_{args.cam}0"
    save_dir = Path(".")
    if args.video:
        video_writer = skvideo.io.FFmpegWriter(
            str(save_dir / f"{args.seed}_{args.cam}.mp4")
        )

    for step in range(args.step, args.step + 100):
        filename = path / f"{args.seed:04d}_{step:05d}.pkl"
        filename_infos = path.parent.parent / "episodes_info" / f"{args.seed:04d}.pkl"
        if not filename.exists():
            if step == args.step:
                print(f"File {filename} not found.")
            break

        real_rgb, gripper_pos, cube_pos = load_data(filename, filename_infos, cam_name)
        sim_rgb = generate_sim_im(env, gripper_pos, cube_pos, cam_name)

        if args.blend:
            pil_sim = Image.fromarray(sim_rgb)
            pil_real = Image.fromarray(real_rgb)

            pil_blend = Image.blend(pil_sim, pil_real, 0.5)
            pil_res = pil_blend.resize(high_res, Image.BILINEAR)
        else:
            sim_rgb = np.asarray(
                Image.fromarray(sim_rgb).resize(high_res, Image.BILINEAR)
            )
            real_rgb = np.asarray(
                Image.fromarray(real_rgb).resize(high_res, Image.BILINEAR)
            )
            rgb = np.hstack((sim_rgb, real_rgb))
            pil_res = Image.fromarray(rgb)

        res = np.asarray(pil_res)
        if args.video:
            video_writer.writeFrame(res)
        else:
            pil_res.save(save_dir / f"{args.seed}.png")
            break

    if args.video:
        video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Align sim and real", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
