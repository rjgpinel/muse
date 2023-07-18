import argparse
import csv
import os
import requests
import zipfile
import shutil

from pathlib import Path
from muse.core import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Download kth-kyberge-uiuc textures script", add_help=False
    )
    return parser


def main(args):
    output_dir = utils.assets_dir() / "textures" / "kth-kyberge-uiuc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # url_download = (
    #     "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    # )

    # print("Downloading tar file...")
    # # setting the default header, else the server does not allow the download
    # headers = {"User-Agent": "Mozilla/5.0"}
    # request = requests.get(url_download, headers=headers)
    # tar_path = output_dir / "dtd-r1.0.1.tar.gz"
    # with open(str(tar_path), "wb") as f:
    #     f.write(request.content)

    zip_path = output_dir / "Splited.zip"
    with zipfile.ZipFile(str(zip_path)) as zip_f:
        zip_f.extractall(path=str(output_dir))

    textures_list = output_dir.glob("**/*.jpg")

    for i, texture_filename in enumerate(textures_list):
        texture_filename.rename(str(output_dir / f"{i}.jpg"))

    zip_path.unlink()
    shutil.rmtree(str(output_dir / "train"))
    shutil.rmtree(str(output_dir / "valid"))
    print(f"Done downloading textures, saved in {str(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Download kth-kyberge-uiuc textures script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
