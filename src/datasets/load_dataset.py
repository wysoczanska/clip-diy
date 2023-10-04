"""
CLIP-DIY: based on COMUS codebase: https://github.com/zadaianchuk/comus
author: Monika Wysoczanska, Warsaw University of Technology
"""


import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data.distributed import DistributedSampler
from src.datasets.coco import COCOSegmentationwithMaskWrapper, COCOSegmentationWrapper, COCOSegmentationDatasetSaliencyFull
from src.datasets.pascal_voc import PascalVOCSegmentation, PascalVOCSegmentationWrapper, \
    PascalVOCSegmentationSaliencyFull
from src.utils import collate_fn_masks, collate_fn_masks_and_sal


def load_data(cfg, split, normalize=False, distributed=False, saliency=False, instances=False):
    if normalize:
        transforms_list = [
        A.Resize(cfg.img_size[0], cfg.img_size[1], interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    else:
        transforms_list = [
            A.Resize(cfg.img_size[0], cfg.img_size[1], interpolation=1),
            ToTensorV2(),
        ]

    transform = A.Compose(transforms_list, additional_targets={"pseudo_mask": "mask"})
    if cfg.dataset.name in ["pascal"]:
        dataset_raw = PascalVOCSegmentation(split=split, root=cfg.dataset.root, transform=None)

        if saliency:
            dataset = PascalVOCSegmentationSaliencyFull(
                split=split, root=cfg.dataset.root, transform=transform, sal_dir=cfg.dataset.sal_dir
            )
            collate_fn = collate_fn_masks_and_sal
        else:
            dataset = PascalVOCSegmentationWrapper(
                split=split, root=cfg.dataset.root, transform=transform
            )
            collate_fn = collate_fn_masks
        n_classes = 21

    elif cfg.dataset.name in ["coco"]:
        dataset_raw = COCOSegmentationwithMaskWrapper(
            root=cfg.dataset.root,
            idx_dir=cfg.dataset.idx_dir,
            split=split,
            cat_list=cfg.dataset.cat_list,
            transform=None
        )
        if saliency:
            dataset = COCOSegmentationDatasetSaliencyFull(root=cfg.dataset.root,
                idx_dir=cfg.dataset.idx_dir,
                split=split,
                cat_list=cfg.dataset.cat_list,
                transform=transform,
                sal_dir=cfg.dataset.sal_dir
            )
            collate_fn = collate_fn_masks_and_sal

        else:
            dataset = COCOSegmentationWrapper(
                root=cfg.dataset.root,
                idx_dir=cfg.dataset.idx_dir,
                split=split,
                cat_list=cfg.dataset.cat_list,
                transform=transform,
            )
            collate_fn = collate_fn_masks

        n_classes = 81
    else:
        raise ValueError(f"{cfg.dataset.name} is not valid. Pick from 'coco' or 'pascal'.")

    if distributed:
        loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.experiment.batch_size,
        shuffle=False,
        num_workers=cfg.experiment.num_workers,
        collate_fn=collate_fn,
        sampler=DistributedSampler(dataset)
    )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.experiment.batch_size,
            shuffle=False,
            num_workers=cfg.experiment.num_workers,
            collate_fn=collate_fn,
        )

    return loader, dataset_raw, n_classes