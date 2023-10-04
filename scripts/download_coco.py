"""Script to download COCO 2017. Source: https://github.com/zadaianchuk/comus"""
import argparse
import os
import zipfile

import wget

IMAGE_URL = "http://images.cocodataset.org/zips"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations"

SPLITS_TO_SUFFIX = {"train": "train2017", "validation": "val2017"}
SPLITS_TO_INSTANCES_FILENAME = {
    "train": "instances_train2017.json",
    "validation": "instances_val2017.json",
}


parser = argparse.ArgumentParser("Generate sharded dataset from original COCO data.")
parser.add_argument(
    "--split",
    default="train",
    choices=list(SPLITS_TO_SUFFIX),
    help="Which splits to write",
)
parser.add_argument(
    "--download-dir",
    default="./data/COCO",
    help="Directory where COCO images and annotations are downloaded to",
)


def download_zip_and_extract(url, dest_dir):
    print(f"Downloading {url} to {dest_dir}")
    file = wget.download(url, out=dest_dir)
    print(f"\nExtracting {file} to {dest_dir}")
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    os.remove(file)


def get_coco_images(data_dir, split):
    assert split in SPLITS_TO_SUFFIX
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    download_zip_and_extract(f"{IMAGE_URL}/{SPLITS_TO_SUFFIX[split]}.zip", image_dir)
    return image_dir


def get_coco_annotations(data_dir, annotation_file="annotations_trainval2017"):
    os.makedirs(data_dir, exist_ok=True)
    download_zip_and_extract(f"{ANNOTATIONS_URL}/{annotation_file}.zip", data_dir)
    return data_dir


if __name__ == "__main__":
    args = parser.parse_args()

    image_dir = os.path.join(args.download_dir, "images", SPLITS_TO_SUFFIX[args.split])
    if not os.path.exists(image_dir):
        get_coco_images(args.download_dir, args.split)
        assert os.path.exists(image_dir)

    if args.split in SPLITS_TO_INSTANCES_FILENAME:
        annotations_dir = os.path.join(args.download_dir, "annotations")
        annotations_file = os.path.join(annotations_dir, SPLITS_TO_INSTANCES_FILENAME[args.split])
        if not os.path.exists(annotations_file):
            get_coco_annotations(args.download_dir)
        assert os.path.exists(annotations_file)
