# from https://github.com/pytorch/vision/blob/main/torchvision/datasets/voc.py
import argparse
import os
import tarfile

from torchvision.datasets.utils import download_url

DATASET_YEAR_DICT = {
    "2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "base_dir": "VOCdevkit/VOC2012",
    }
}

parser = argparse.ArgumentParser("Download original PASCAL data.")

parser.add_argument(
    "--download-dir",
    default="./data/PASCAL_VOCL",
    help="Directory where COCO images and annotations are downloaded to",
)


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)


if __name__ == "__main__":
    args = parser.parse_args()

    download_extract(
        DATASET_YEAR_DICT["2012"]["url"],
        args.download_dir,
        DATASET_YEAR_DICT["2012"]["filename"],
        DATASET_YEAR_DICT["2012"]["md5"],
    )
