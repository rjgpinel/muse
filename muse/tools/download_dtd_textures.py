import argparse
import csv
import os
import requests
import tarfile

from io import BytesIO
from pathlib import Path
from PIL import Image
from muse.core import utils


def get_args_parser():
    parser = argparse.ArgumentParser("Download dtd textures script", add_help=False)
    return parser


def main(args):
    output_dir = utils.assets_dir() / "textures" / "dtd"
    output_dir.mkdir(parents=True, exist_ok=True)

    url_download = (
        "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    )

    # print("Downloading tar file...")
    # # setting the default header, else the server does not allow the download
    # headers = {"User-Agent": "Mozilla/5.0"}
    # request = requests.get(url_download, headers=headers)
    # tar_path = output_dir / "dtd-r1.0.1.tar.gz"
    # with open(str(tar_path), "wb") as f:
    #     f.write(request.content)

    # textures_tar = tarfile.open(str(tar_path))
    # textures_tar.extractall(str(output_dir))
    # textures_tar.close()

    textures_list = output_dir.glob("**/*.jpg")

    for texture_filename in textures_list:
        texture_filename.rename(str(output_dir / texture_filename.name))

    print(f"Done downloading textures, saved in {str(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Download dtd textures script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
