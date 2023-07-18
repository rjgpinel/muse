import argparse

import skvideo.io
import pickle as pkl

from pathlib import Path
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("BC generate video script", add_help=False)
    parser.add_argument("--demo-path", default="", type=str)
    parser.add_argument("--episode", default=20, type=int)
    parser.add_argument("--cam-name", default="charlie_camera0", type=str)
    parser.add_argument("--output-path", default="", type=str)
    return parser


def create_video(demo_path, episode, cam_name, output_path):
    pkl_list = sorted(list(demo_path.glob(f"{episode:04d}_*.pkl")))
    assert len(pkl_list) > 0, "no steps for episode {episode}"
    output_path.mkdir(parents=True, exist_ok=True)
    video_writer = skvideo.io.FFmpegWriter(str(output_path / f"{episode:04d}.mp4"))

    for step_path in tqdm(pkl_list):
        with open(str(step_path), "rb") as f:
            step = pkl.load(f)

        obs, _ = step
        rgb = obs[f"rgb_{cam_name}"]
        video_writer.writeFrame(rgb)

    video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BC generate video script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    demo_path = Path(args.demo_path)
    output_path = Path(args.output_path)
    assert demo_path.exists(), "demo_path does not exist"
    create_video(demo_path, args.episode, args.cam_name, output_path)
