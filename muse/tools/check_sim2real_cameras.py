import argparse
import pickle as pkl
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
from muse.envs.utils import realsense_resize_crop


def get_args_parser():
    parser = argparse.ArgumentParser("Check sim2real cameras", add_help=False)
    parser.add_argument(
        "--pickle-dir", default="/home/rgarciap/Desktop/camera_check/", type=str
    )
    return parser


def main(args):
    pkl_dir = Path(args.pickle_dir)

    for pkl_filename in pkl_dir.glob("*.pkl"):
        if "cropped" in str(pkl_filename):
            continue

        seed = pkl_filename.name[:-4]
        with open(str(pkl_filename), "rb") as f:
            obs_sim, obs_real = pkl.load(f)

        for cam_name in ["charlie_camera", "bravo_camera"]:

            obs_real[f"rgb_{cam_name}"] = realsense_resize_crop(
                obs_real[f"rgb_{cam_name}"], im_type="rgb"
            )
            obs_real[f"depth_{cam_name}"] = realsense_resize_crop(
                obs_real[f"depth_{cam_name}"], im_type="depth"
            )

            depth = np.concatenate(
                [obs_sim[f"depth_{cam_name}0"], obs_real[f"depth_{cam_name}"]], axis=1
            )
            rgb = np.concatenate(
                [obs_sim[f"rgb_{cam_name}0"], obs_real[f"rgb_{cam_name}"]], axis=1
            )

            diff = obs_sim[f"depth_{cam_name}0"] - obs_real[f"depth_{cam_name}"]

            with open(str(pkl_dir / f"{seed}_cropped.pkl"), "wb") as f:
                pkl.dump([obs_sim, obs_real], f)

            plt.subplot(211)
            plt.imshow(depth)
            plt.subplot(212)
            plt.imshow(rgb)
            plt.show()

            plt.imshow(diff)
            plt.show()

            mpimg.imsave(f"{seed}_{cam_name}_depths.png", depth)
            mpimg.imsave(f"{seed}_{cam_name}_rgb.png", rgb)
            mpimg.imsave(f"{seed}_{cam_name}_diff.png", diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Check sim2real cameras", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
