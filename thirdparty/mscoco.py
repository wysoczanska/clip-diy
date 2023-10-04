"""MSCOCO Semantic Segmentation."""
# ------------------------------------------------------------------------
# Semantic Segmentation on PyTorch (https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
# Copyright (c) https://github.com/Tramac. All Rights Reserved.
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# MSCOCO is adapted from in addition to COCO PASCAL classes, any cat_list is supported
# for full COCO validation dataset was not filtered.

import logging
import os
import pickle

import numpy as np
import pycocotools.mask
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import trange

from thirdparty.utils import mkdir_if_missing

log = logging.getLogger(__name__)

PASCAL_VOC_CLASSES = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

COCO_CLASSES_FULL = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]
CAT_LISTS = {"pascal": PASCAL_VOC_CLASSES, "coco_full": COCO_CLASSES_FULL}


class COCOSegmentation(torch.utils.data.Dataset):
    """COCO Semantic Segmentation Dataset with original COCO and PASCAL VOC categories.

    Instances of the same class are merged together.
    Parameters
    ----------
    root : string
        Path to COCO folder.
    idx_dir: string
        Path to store new indexes and other info.
    cat_list: string
        'coco_full' (80 categories) or 'coco_pascal' (20 categories)
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    """

    def __init__(self, root, idx_dir, cat_list, split, transform=None):
        super(COCOSegmentation, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.cat_list = CAT_LISTS[cat_list]
        self.num_class = len(self.cat_list)
        assert cat_list in ["coco_full", "pascal"]
        self.cat_list_name = cat_list
        self.idx_dir = idx_dir
        idx_dir = os.path.join(idx_dir, "annotations")
        mkdir_if_missing(idx_dir)
        if split == "train":
            log.info("Using train set")
            ann_file = os.path.join(root, "annotations/instances_train2017.json")
            ids_file = os.path.join(idx_dir, f"{self.cat_list_name}_train_ids.mx")
            self.root = os.path.join(root, "images/train2017")
            self.split = split
        else:
            log.info("Using val set")
            ann_file = os.path.join(root, "annotations/instances_val2017.json")
            ids_file = os.path.join(idx_dir, f"full_{self.cat_list_name}_val_ids.mx")
            self.root = os.path.join(root, "images/val2017")
        self.coco = COCO(ann_file)
        if os.path.exists(ids_file):
            with open(ids_file, "rb") as file_:
                self.ids = pickle.load(file_)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

    def get_mask(self, index):
        img_id = self.ids[index]
        img_metadata = self.coco.loadImgs(img_id)[0]
        cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        mask = self._gen_seg_mask(cocotarget, img_metadata["height"], img_metadata["width"])
        mask = np.array(mask).astype("int32")
        return mask

    def get_shape(self, img_id):
        img_metadata = self.coco.loadImgs(img_id)[0]
        return img_metadata["height"], img_metadata["width"]

    def get_image(self, index):
        img_id = int(self.ids[index])
        img_metadata = self.coco.loadImgs(img_id)[0]
        path = img_metadata["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = np.array(img)
        return img

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = np.array(img)
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = self._gen_seg_mask(cocotarget, img_metadata["height"], img_metadata["width"])
        mask = np.array(mask).astype("int32")
        return img, mask

    def __len__(self):
        return len(self.ids)

    def _gen_seg_mask(self, target, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for instance in target:
            rle = pycocotools.mask.frPyObjects(instance["segmentation"], height, width)
            instance_mask = pycocotools.mask.decode(rle)
            cat = instance["category_id"]
            if cat in self.cat_list:
                valid_cat = self.cat_list.index(cat)
            else:
                continue
            if len(instance_mask.shape) < 3:
                mask[:, :] += (mask == 0) * (instance_mask * valid_cat)
            else:
                mask[:, :] += (mask == 0) * (
                    ((np.sum(instance_mask, axis=2)) > 0) * valid_cat
                ).astype(np.uint8)
        return mask

    def _convert_to_segmentation_mask_lables(self, mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(self.cat_list)), dtype=np.int32)
        for label_index, _ in enumerate(self.cat_list):
            bin_mask = (mask == label_index).astype(np.int32)
            segmentation_mask[:, :, label_index] = bin_mask
        return segmentation_mask

    def _preprocess(self, ids, ids_file):
        log.info(
            "Preprocessing mask, this will take a while."
            + "But don't worry, it only run once for each split."
        )
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata["height"], img_metadata["width"])
            # more than 1k pixels
            if (self.cat_list_name == "full_coco") and (self.split == "val"):
                new_ids.append(img_id)
            elif (mask > 0).sum() >= 1000:
                new_ids.append(img_id)

            tbar.set_description(f"Doing: {i}/{len(ids)}, got {len(new_ids)} qualified images")

        log.info(f"Found number of qualified images: {len(new_ids)}")
        with open(ids_file, "wb") as file_:
            pickle.dump(new_ids, file_)
        return new_ids

    @property
    def classes(self):
        """Category names."""
        return (
            "background",
            "airplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorcycle",
            "person",
            "potted-plant",
            "sheep",
            "sofa",
            "train",
            "tv",
        )
