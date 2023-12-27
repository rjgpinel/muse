import argparse
from pathlib import Path
import numpy as np
import time

import muse.envs
from muse.core import constants
import gym

import matplotlib.pyplot as plt
from PIL import Image

import skvideo.io


def get_args_parser():
    parser = argparse.ArgumentParser("Run MUSE environments", add_help=False)
    parser.add_argument("--env", default="Pick-v0", type=str)
    parser.add_argument("--render", dest="render", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--video-dir", default="/tmp/demos", type=str)
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--num-textures", default=None, type=int)
    parser.add_argument("--textures-pack", default="bop", type=str)
    parser.add_argument("--cam-xyz-noise", default=0.0, type=float)
    parser.add_argument("--light-setup", default="default", type=str)
    parser.add_argument("--num-distractors", default=0, type=int)
    parser.set_defaults(viewer=True, render=False)
    return parser


def plot_actions(vel, max_vel=constants.MAX_TOOL_VELOCITY[0]):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for i, label in enumerate(["x", "y", "z", "grip"]):
        if i < 3:
            ax = axs[0]
        else:
            ax = axs[1]
        ax.plot(np.arange(vel.shape[0]), vel[:, i], label=label)
        ax.set_xlabel("t")
        ax.set_ylabel(f"{label}_vel")
        if label != "grip":
            ax.set_ylim(-max_vel, max_vel)
    for i in range(2):
        axs[i].legend()
    plt.show()


def main(args):
    if args.render:
        args.viewer = False

    env = gym.make(
        args.env,
        viewer=args.viewer,
        cam_render=args.render,
        num_textures=args.num_textures,
        num_distractors=args.num_distractors,
        textures_pack=args.textures_pack,
        cam_xyz_noise=args.cam_xyz_noise,
        light_setup=args.light_setup,
    )
    env.seed(args.seed)

    t0 = time.time()
    obs = env.reset()
    if args.viewer:
        env.unwrapped.render()
    if args.render:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        video_writer = skvideo.io.FFmpegWriter(str(video_dir / f"{args.seed}.mp4"))

    agent = env.unwrapped.oracle()
    actions = []
    seed = args.seed
    seed_max = args.seed + args.episodes
    done = False

    i = 0 
    while True:
        action = agent.get_action(obs)
        if action is None or done:
            if info["success"]:
                text = "success"
            else:
                text = "failure"
            print(f"{seed}: {text} - {time.time()-t0} ")
            seed += 1
            if seed >= seed_max:
                break
            if args.render:
                video_writer.close()
                video_writer = skvideo.io.FFmpegWriter(str(video_dir / f"{seed}.mp4"))
            env.seed(seed)
            obs = env.reset()
            agent = env.unwrapped.oracle()
            action = agent.get_action(obs)
            i = 0

        actions.append(np.hstack([v for v in action.values()]))
        obs, reward, done, info = env.step(action)
        i+=1
        
        if i % 10 == 0:
            env.scene.modder.apply(env.workspace)
            # env.rand_visual()

        if args.viewer:
            env.unwrapped.render()

        if args.render:
            im = np.hstack((obs["rgb_bravo_camera"], obs["rgb_charlie_camera"]))
            video_writer.writeFrame(im)

    actions = np.stack(actions)
    # plot_actions(actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run MUSE environments", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
