import argparse
import csv
import os
import requests
import zipfile
import random

from io import BytesIO
from pathlib import Path
from PIL import Image
from muse.core import utils

TEXTURES_BLACKLIST = [
    "sign",
    "roadlines",
    "manhole",
    "backdrop",
    "foliage",
    "TreeEnd",
    "TreeStump",
    "3DBread",
    "3DApple",
    "FlowerSet",
    "FoodSteps",
    "PineNeedles",
    "Grate",
    "PavingEdge",
    "Painting",
    "RockBrush",
    "WrinklesBrush",
    "Sticker",
    "3DRock",
    "PaintedWood001",
    "PaintedWood003",
    "PaintedWood002",
    "PaintedWood004",
    "PaintedWood005",
    "Carpet002",
    "SurfaceImperfections001",
    "SurfaceImperfections002",
    "SurfaceImperfections003",
    "SurfaceImperfections004",
    "SurfaceImperfections005",
    "SurfaceImperfections006",
    "SurfaceImperfections007",
    "SurfaceImperfections008",
    "SurfaceImperfections009",
    "SurfaceImperfections010",
    "SurfaceImperfections011",
    "SurfaceImperfections012",
    "SurfaceImperfections013",
    "Scratches001",
    "Scratches002",
    "Scratches003",
    "Scratches004",
    "Scratches005",
    "Tiles001",
    "DiamondPlate004",
    "Smear005",
    "Smear006",
    "Smear007",
    "Fingerprints004",
    "Fabric019",
    "Fabric036",
    "Fabric039",
    "Clay004",
    "Concrete006",
    "AcousticFoam001",
    "AcousticFoam002",
    "AcousticFoam003",
    "SheetMetal002",
    "Porcelain001",
    "Porcelain002",
    "Porcelain003",
    "Paper003",
    "OfficeCeiling001",
    "OfficeCeiling006",
    "Tiles036",
    "Payment",
    "Tape003",
]


def get_args_parser():
    parser = argparse.ArgumentParser("Download textures script", add_help=False)
    parser.add_argument("--crop-size", default=256, type=int)
    parser.add_argument(
        "--n", default=None, type=int, help="number of textures to download"
    )
    parser.add_argument("--seed", default=0, type=int)
    return parser


def main(args):
    output_dir = utils.assets_dir() / "textures" / "bop"
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_size = args.crop_size
    compress = crop_size is not None

    # download the csv file, which contains all the download links
    csv_path = output_dir / "full_info.csv"
    csv_url = "https://cc0textures.com/api/v1/downloads_csv"
    # setting the default header, else the server does not allow the download
    headers = {"User-Agent": "Mozilla/5.0"}
    request = requests.get(csv_url, headers=headers)
    with open(str(csv_path), "wb") as f:
        f.write(request.content)

    # extract the download links with the asset name
    textures_url = {}
    with open(str(csv_path), "r") as csv_f:
        csv_reader = csv.DictReader(csv_f, delimiter=",")
        for line in csv_reader:
            if line["Filetype"] == "zip" and line["DownloadAttribute"] == "1K-JPG":
                textures_url[line["AssetID"]] = line["PrettyDownloadLink"]

    textures_key = list(textures_url.keys())
    if args.n:
        random.seed(args.seed)
        random.shuffle(textures_key)
        textures_key = textures_key[: args.n]

    # download each asset and create a folder for it (unpacking + deleting the zip included)
    for idx, asset in enumerate(textures_key):
        texture_url = textures_url[asset]
        blacklisted = False
        for blacklisted_asset in TEXTURES_BLACKLIST:
            if asset.lower().startswith(blacklisted_asset.lower()):
                blacklisted = True
                break
        if not blacklisted:
            print(f"Download asset: {asset} of {idx}/{len(textures_key)}")
            response = requests.get(texture_url, headers=headers)
            f = BytesIO(response.content)
            try:
                with zipfile.ZipFile(f) as zip_f:
                    zip_f.extract(f"{asset}_1K_Color.jpg", str(output_dir))
                # Process compress image
                texture_path = str(output_dir / f"{asset}_1K_Color.jpg")
                texture_im = Image.open(texture_path)
                if compress:
                    biggest_size = min(texture_im.size)
                    ratio = crop_size / float(biggest_size)
                    if texture_im.size[0] > texture_im.size[1]:
                        crop_w = crop_size * ratio
                        crop_h = crop_size
                    else:
                        crop_w = crop_size
                        crop_h = crop_size * ratio
                    texture_im = texture_im.resize(
                        (
                            int(texture_im.size[0] * ratio),
                            int(texture_im.size[1] * ratio),
                        )
                    )
                    texture_im = texture_im.crop((0, 0, crop_size, crop_size))

                # Remove JPG file
                (output_dir / f"{asset}_1K_Color.jpg").unlink()
                texture_im.save(str(output_dir / f"{asset}_Color.png"))

            except (IOError, zipfile.BadZipfile) as e:
                print(f"Bad zip file given as input {asset}. Skipping")

    print(f"Done downloading textures, saved in {str(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Download textures script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
