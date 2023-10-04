"""
CLIP-DIY: based on COMUS codebase: https://github.com/zadaianchuk/comus
author: Monika Wysoczanska, Warsaw University of Technology
"""

import logging
import os
import numpy as np
import torch

from thirdparty.mscoco import COCOSegmentation

import torchvision.transforms as T
from torchvision.transforms.functional import resize
from PIL import Image

log = logging.getLogger(__name__)

COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


COCO_CLASS_NAMES_ALL = ['background'] + COCO_CLASS_NAMES


class COCOSegmentationWrapper(COCOSegmentation):
    def __getitem__(self, index):
        img_id = self.ids[index]
        image = super().get_image(index)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return img_id, image, torch.zeros_like(image)


class COCOSegmentationwithMaskWrapper(COCOSegmentation):
    def __getitem__(self, index):
        img_id = self.ids[index]
        image = super().get_image(index)
        mask = super().get_mask(index)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            mask, image = transformed["mask"], transformed["image"]
        return img_id, image, mask

    def get_mask(self, index):
        mask = super().get_mask(index)
        return np.array(resize(Image.fromarray(mask), size=448, interpolation=T.InterpolationMode.NEAREST))


class COCOSegmentationDatasetSaliencyFull(COCOSegmentation):
    def __init__(self, root, idx_dir, cat_list, split, transform, sal_dir):
        super().__init__(root=root, idx_dir=idx_dir, cat_list=cat_list, split=split, transform=transform)
        sal_dir = sal_dir
        self.saliency = [os.path.join(sal_dir, f"{x:012}.npy") for x in self.ids]
        self.name = 'coco'

    def __getitem__(self, index):
        img_id = self.ids[index]
        image = super().get_image(index)

        with open(self.saliency[index], 'rb') as f:
            saliency = np.load(f)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            saliency = self.transform(image=saliency)["image"]

        return img_id, image, None, saliency
