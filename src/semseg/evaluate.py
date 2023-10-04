"""
CLIP-DIY: based on COMUS codebase: https://github.com/zadaianchuk/comus
author: Monika Wysoczanska, Warsaw University of Technology
"""

from tqdm import tqdm
import os
import cv2
import pandas as pd
from joblib import Parallel, delayed
from src.datasets.coco import COCO_CLASS_NAMES_ALL
from src.datasets.pascal_voc import VOC_CLASSES
from thirdparty.utils import get_iou
import logging

import numpy as np

N_JOBS = 8

log = logging.getLogger(__name__)


def compute_predictions(model, loader):
    predictions = []
    img_ids_all = []
    for img_ids, imgs, _ in tqdm(loader, "Masks inference"):
        inputs = imgs.cuda()
        outputs = model(inputs)["out"].argmax(dim=1).cpu().numpy()
        predictions.append(outputs)
        img_ids_all.append(img_ids)
    img_ids = np.concatenate(img_ids_all)
    predictions = np.concatenate(predictions, axis=0)
    return predictions, img_ids


def combine_predictions(predictions, img_ids, dataset_raw, max_size=700, ignore_label=None):
    # Load all pixel embeddings
    all_pixel_predictions = np.zeros((len(dataset_raw) * max_size * max_size), dtype=np.float32)
    all_gt = np.zeros((len(dataset_raw) * max_size * max_size), dtype=np.float32)
    offset_ = 0
    for idx in range(len(dataset_raw)):
        gt_mask = dataset_raw.get_mask(idx)
        prediction = get_prediction(predictions, idx, dataset_raw, img_ids, gt_mask.shape)

        valid = gt_mask != 255

        if ignore_label is not None:
            not_ignored = gt_mask != ignore_label
            valid = valid & not_ignored
        n_valid = np.sum(valid)
        all_gt[offset_ : offset_ + n_valid] = gt_mask[valid]

        # Possibly reshape embedding to match gt.
        if prediction.shape != gt_mask.shape:
            prediction = cv2.resize(prediction, gt_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
        all_pixel_predictions[offset_ : offset_ + n_valid] = prediction[valid]
        all_gt[offset_ : offset_ + n_valid] = gt_mask[valid]

        # Update offset_
        offset_ += n_valid
    # All pixels, all ground-truth
    all_pixel_predictions = all_pixel_predictions[:offset_]
    all_gt = all_gt[:offset_]
    log.info(f"Found {all_gt.shape[0]} valid pixels.")
    return all_pixel_predictions, all_gt


def evaluate_predictions(
    pixel_predictions,
    pixel_gt_labels,
    n_classes,
    dataset_results_dir,
    results_file_name,
    verbose=True,
):
    """Evaluation of flatten predictions.

    Given two vectors of flatten dense GT lables
    and corresponding predictions from unsupervised semantic segmentation with clusters ID
    computes mIoU using provided matching.

    Args:
        pixel_predictions (np.array, Px1): all the pixel predictions (with clustering ID as lables)
        pixel_gt_labels (np.array, Px1): all the pixel lables from original images
        n_classes (int): Number of GT classes
        dataset_results_dir (str): dir path to store the results
        results_file_name (str): file name to store the results

    Returns:
        eval_result (dict): dict with mIoU and PA  as well as per category IoU
    """

    log.info("Evaluation of semantic segmentation")

    iou = Parallel(n_jobs=N_JOBS, backend="multiprocessing")(
        delayed(get_iou)(pixel_predictions, pixel_gt_labels, i_part, i_part) for i_part in range(n_classes)
    )
    iou = np.array(iou)
    pixel_accuracy = get_pixel_accuracy(pixel_predictions, pixel_gt_labels)

    eval_result = {
        "per_class": iou,
        "mIoU": np.mean(iou),
        "PA": pixel_accuracy,
    }

    data_path = os.path.join(dataset_results_dir, f"results.npz")
    np.savez(data_path, **eval_result)

    log.info("Pixel Accuracy is %.2f" % (100 * eval_result["PA"]))
    log.info("Mean IoU is %.2f" % (100 * eval_result["mIoU"]))

    if n_classes == 21:
        class_names = VOC_CLASSES
    elif n_classes == 81:
        class_names = COCO_CLASS_NAMES_ALL
    else:
        raise ValueError(f"Not valid number of classes: {n_classes}.")
    data_f = {
        "Classes": ["mIoU", "PA"] + class_names,
        "IoU": [eval_result["mIoU"], eval_result["PA"]] + eval_result["per_class"].tolist(),
    }
    data_path = os.path.join(dataset_results_dir, f"{results_file_name}")
    log.info(dataset_results_dir)
    log.info(data_path)
    pd.DataFrame(data_f).to_csv(data_path)
    if verbose:
        for i_part in range(n_classes):
            log.info(f"Class {class_names[i_part]} has IoU {100 * iou[i_part]:.2f}")
    return eval_result


def one_label_mask(sal_mask, label):
    mask = np.zeros_like(sal_mask).astype(np.int32)
    mask[sal_mask.astype(np.bool)] = label + 1
    return mask


def get_prediction(predictions, idx, dataset_raw, img_ids, shape):
    img_ids = list(img_ids)
    if dataset_raw.ids[idx] in img_ids:
        return predictions[img_ids.index(dataset_raw.ids[idx])]
    else:
        return np.zeros(shape)


def get_pixel_accuracy(flat_preds, flat_targets):
    tp = (flat_preds == flat_targets).sum()
    return float(tp) / flat_preds.shape[0]
