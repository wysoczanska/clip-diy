"""
CLIP-DIY: based on COMUS codebase: https://github.com/zadaianchuk/comus
author: Monika Wysoczanska, Warsaw University of Technology
"""

import logging
import os
from typing import List
from torchvision.transforms.functional import resize
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset

log = logging.getLogger(__name__)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

VOC_CLASSES = [
    "background",
    "aeroplane",
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
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class PascalVOCSegmentation(VisionDataset):
    classes = VOC_CLASSES
    _SPLITS_DIR = "Segmentation"
    _TARGET_FILE_EXT = ".png"

    def __init__(self, root, split="val", transform=None):
        self.transform = transform
        self.root = root
        self.year = 2012
        self._TARGET_DIR = "SegmentationClass"
        self.name = 'pascal'

        base_dir = "VOCdevkit/VOC2012"
        voc_root = os.path.join(self.root, base_dir)

        if not os.path.isdir(voc_root):
            raise RuntimeError(f"{voc_root} Dataset not found or corrupted.")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, split.rstrip("\n") + ".txt")
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, f"{x}.jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]
        self.file_names = file_names

    @property
    def ids(self):
        return self.file_names

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        """Converts a mask from the Pascal VOC format to the format required by AutoAlbument."""
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)

        return segmentation_mask

    @staticmethod
    def _convert_to_segmentation_mask_lables(mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, _ in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label_index, axis=-1).astype(float)
        return segmentation_mask

    def __len__(self) -> int:
        return len(self.images)

    @property
    def masks(self) -> List[str]:
        return self.targets

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1)

        return index, image, mask

    def get_shape(self, img_id):
        image = cv2.imread(img_id)
        return image.shape[:2]

    def get_mask(self, index):
        mask = Image.open(self.masks[index])

        # fix for the eval
        mask = resize(mask, size=448, interpolation=T.InterpolationMode.NEAREST)
        return np.array(mask)


class PascalVOCSegmentationWrapper(PascalVOCSegmentation):
    def __getitem__(self, index):
        _, image, mask = super().__getitem__(index)
        return self.ids[index], image, mask


class PascalVOCSegmentationSaliencyFull(PascalVOCSegmentation):
    def __init__(self, root, split, transform=None, sal_dir=None):
        super().__init__(root=root, split=split, transform=transform)
        self.saliency = [os.path.join(sal_dir, f"{x}.npy") for x in self.file_names]

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with open(self.saliency[index], 'rb') as f:
            saliency = np.load(f)

        if self.transform is not None:
            image = self.transform(image=image)['image']
            saliency = self.transform(image=saliency)['image']

        return self.ids[index], image, None, saliency

