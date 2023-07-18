import argparse
import gym
import time
import numpy as np
import muse.envs

import matplotlib as mpl
import matplotlib.pyplot as plt

from colorsys import rgb_to_hsv, hsv_to_rgb
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from muse.core import constants


def get_args_parser():
    parser = argparse.ArgumentParser("Record DR instances", add_help=False)
    parser.add_argument("--env", default="DR-Pick-v0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output-dir", default="/tmp/instances/", type=str)
    parser.add_argument("--instances", default=20, type=int)
    parser.add_argument("--num-textures", default=None, type=int)
    parser.add_argument("--camera", default="charlie_camera", type=str)
    parser.add_argument("--num-distractors", default=0, type=int)
    return parser


# def plot_range(
#     output_path,
#     mean,
#     var,
#     vmax=60,
#     vmin=30,
#     cmap=mpl.cm.cool,
#     label="FOV",
#     line_width=0.0001,
#     color_mean=[0, 0, 1, 1],
#     color_var=[0, 1, 0, 0.4],
# ):

#     mpl.rcParams.update({"font.size": 22})
#     fig, ax = plt.subplots(figsize=(6, 2))
#     fig.subplots_adjust(bottom=0.5)

#     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

#     fig.colorbar(
#         mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#         cax=ax,
#         orientation="horizontal",
#         label=label,
#     )
#     mean_v = plt.Rectangle((mean, 0.0), line_width, 1.0, color=color_mean)
#     var_v = plt.Rectangle((mean - var, 0.0), var * 2, 1.0, color=color_var, alpha=0.5)
#     ax.add_patch(mean_v)
#     ax.add_patch(var_v)
#     plt.tight_layout()
#     plt.savefig(f"{str(output_path)}")


# def hsv_wheel(
#     output_path,
#     hsv_rand_factor,
#     cube_0_rgb=[0.74, 0.13, 0.10],
#     cube_1_rgb=[0, 0.48, 0.36],
#     cube_2_rgb=[0.93, 0.86, 0.16],
# ):

#     colors = ["red", "green", "yellow"]
#     cube_0_hsv = rgb_to_hsv(*cube_0_rgb)
#     cube_1_hsv = rgb_to_hsv(*cube_1_rgb)
#     cube_2_hsv = rgb_to_hsv(*cube_2_rgb)

#     h_var, s_var, v_var = constants.HSV_RAND_VAR
#     h_var, s_var, v_var = (
#         h_var * hsv_rand_factor,
#         s_var * hsv_rand_factor,
#         v_var * hsv_rand_factor,
#     )

#     azimuths = np.arange(0, 361, 1)
#     zeniths = np.arange(40, 70, 1)
#     values = azimuths * np.ones((30, 361))
#     mpl.rcParams.update({"font.size": 13})
#     fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(4, 5))
#     ax.pcolormesh(azimuths * np.pi / 180.0, zeniths, values, cmap="hsv", alpha=1)
#     ax.set_yticks([])
#     ax.set_rorigin(-3.5)

#     ax.add_patch(
#         mpl.patches.Rectangle(
#             ((cube_0_hsv[0] * 2 * np.pi - h_var * 2 * np.pi), 40),
#             width=h_var * 4 * np.pi,
#             height=6,
#             alpha=1,
#             color=cube_0_rgb,
#         )
#     )

#     ax.add_patch(
#         mpl.patches.Rectangle(
#             ((cube_1_hsv[0] * 2 * np.pi - h_var * 2 * np.pi), 46),
#             width=h_var * 4 * np.pi,
#             height=6,
#             alpha=1,
#             color=cube_1_rgb,
#         )
#     )

#     ax.add_patch(
#         mpl.patches.Rectangle(
#             ((cube_2_hsv[0] * 2 * np.pi - h_var * 2 * np.pi), 52),
#             width=h_var * 4 * np.pi,
#             height=6,
#             alpha=1,
#             color=cube_2_rgb,
#         )
#     )

#     ax.bar(0, 1).remove()
#     plt.title("Hue")

#     wheel_path = output_path / "wheel.png"
#     plt.savefig(f"{str(wheel_path)}")

#     line_width = 0.012

#     mpl.rcParams.update({"font.size": 22})
#     plot_range(
#         output_path / "cube0_saturation.png",
#         cube_0_hsv[1],
#         s_var,
#         vmax=1,
#         vmin=0,
#         cmap=mpl.cm.gist_yarg,
#         label="Saturation - red cube",
#         line_width=line_width,
#         color_mean=cube_0_rgb,
#         color_var=cube_0_rgb,
#     )

#     plot_range(
#         output_path / "cube1_saturation.png",
#         cube_1_hsv[1],
#         s_var,
#         vmax=1,
#         vmin=0,
#         cmap=mpl.cm.gist_yarg,
#         label="Saturation - green cube",
#         line_width=line_width,
#         color_mean=cube_1_rgb,
#         color_var=cube_1_rgb,
#     )

#     plot_range(
#         output_path / "cube2_saturation.png",
#         cube_2_hsv[1],
#         s_var,
#         vmax=1,
#         vmin=0,
#         cmap=mpl.cm.gist_yarg,
#         label="Saturation - yellow cube",
#         line_width=line_width,
#         color_mean=cube_2_rgb,
#         color_var=cube_2_rgb,
#     )

#     plot_range(
#         output_path / "cube0_value.png",
#         cube_0_hsv[2],
#         v_var,
#         vmax=1,
#         vmin=0,
#         cmap=mpl.cm.gray,
#         label="Value - red cube",
#         line_width=line_width,
#         color_mean=cube_0_rgb,
#         color_var=cube_0_rgb,
#     )

#     plot_range(
#         output_path / "cube1_value.png",
#         cube_1_hsv[2],
#         v_var,
#         vmax=1,
#         vmin=0,
#         cmap=mpl.cm.gray,
#         label="Value - green cube",
#         line_width=line_width,
#         color_mean=cube_1_rgb,
#         color_var=cube_1_rgb,
#     )

#     plot_range(
#         output_path / "cube2_value.png",
#         cube_2_hsv[2],
#         v_var,
#         vmax=1,
#         vmin=0,
#         cmap=mpl.cm.gray,
#         label="Value - yellow cube",
#         line_width=line_width,
#         color_mean=cube_2_rgb,
#         color_var=cube_2_rgb,
#     )


def main(args):
    env = gym.make(
        args.env,
        num_textures=args.num_textures,
        num_distractors=args.num_distractors,
    )
    env.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = env.reset()

    # if args.xyz_cam_rand_factor > 0:
    #     plot_range(
    #         output_dir / "fov.png",
    #         42.5,
    #         constants.CAM_RAND_VAR[2] * args.fov_cam_rand_factor,
    #         vmax=55,
    #         vmin=30,
    #         label="FOV",
    #     )
    # if args.hsv_rand_factor > 0:
    #     hsv_wheel(output_dir, args.hsv_rand_factor)

    # if args.light_rand_factor > 0:
    #     ambient_value, diffuse_value, specular_value = constants.LIGHTING_DEFAULT_PARAMS
    #     ambient_var, diffuse_var, specular_var = constants.LIGHTING_RAND_VAR
    #     line_width = 0.004
    #     plot_range(
    #         output_dir / "ambient_light.png",
    #         ambient_value,
    #         ambient_var * args.light_rand_factor,
    #         vmax=1,
    #         vmin=0,
    #         label="Ambient light",
    #         cmap=mpl.cm.gist_gray,
    #         line_width=line_width,
    #     )
    #     plot_range(
    #         output_dir / "diffuse_light.png",
    #         diffuse_value,
    #         diffuse_var * args.light_rand_factor,
    #         vmax=1,
    #         vmin=0,
    #         label="Diffuse light",
    #         cmap=mpl.cm.gist_gray,
    #         line_width=line_width,
    #     )
    #     plot_range(
    #         output_dir / "specular_light.png",
    #         specular_value,
    #         specular_var * args.light_rand_factor,
    #         vmax=1,
    #         vmin=0,
    #         label="Specular light",
    #         cmap=mpl.cm.gist_gray,
    #         line_width=line_width,
    #     )

    for i in tqdm(range(args.instances)):
        # env.seed(args.seed)
        # env.scene.modder._np_random = np.random.RandomState(0)
        obs = env.reset()
        for camera in ["bravo_camera", "charlie_camera"]:
            im = obs[f"rgb_{camera}"]
            im = Image.fromarray((im).astype(np.uint8))
            im.save(str(output_dir / f"{i}_{camera}.png"))
        if "rgb_ext_camera" in obs.keys():
            im = obs[f"rgb_ext_camera"]
            im = Image.fromarray((im).astype(np.uint8))
            im.save(str(output_dir / f"ext_{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Record DR instances", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
