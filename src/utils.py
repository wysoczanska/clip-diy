"""
CLIP-DIY: based on COMUS codebase: https://github.com/zadaianchuk/comus
author: Monika Wysoczanska, Warsaw University of Technology
"""

import torch


def collate_fn_masks(batch):
    index = []
    images = []
    masks = []

    for sample in batch:
        index.append(sample[0])
        images.append(sample[1])
        masks.append(sample[2])

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    return index, images, masks


def collate_fn_masks_and_sal(batch):
    index = []
    images = []
    sal = []

    for sample in batch:
        index.append(sample[0])
        images.append(sample[1])
        sal.append(sample[3])

    images = torch.stack(images, dim=0)
    sal = torch.stack(sal, dim=0)

    return index, images, None, sal




