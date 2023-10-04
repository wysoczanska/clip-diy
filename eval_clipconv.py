"""
CLIP-DIY: based on COMUS codebase: https://github.com/zadaianchuk/comus
author: Monika Wysoczanska, Warsaw University of Technology
"""


from src.clip_conv import ClipConv
import numpy as np
from src.datasets.coco import COCO_CLASS_NAMES_ALL
from src.datasets.pascal_voc import VOC_CLASSES
from src.datasets.load_dataset import load_data
from src.semseg.evaluate import evaluate_predictions, combine_predictions
import tqdm
import os
import torch
import pickle
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import logging
import argparse
from hydra import compose, initialize

import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
L = 8

log = logging.getLogger(__name__)


def get_predictions(gpu, cfg, world_size, class_names):
    ddp_setup(gpu, world_size, cfg.experiment.port)
    loader, _, _ = load_data(cfg, cfg.dataset.split, False, world_size> 1, cfg.saliency)
    M = ClipConv(cfg, class_names).to(gpu)
    model = DDP(M, device_ids=[gpu])
    model.eval()

    masks = []
    masks_raw = []
    img_ids = []
    for batch in tqdm.tqdm(loader):
        out = model.forward(batch[1])
        masks_raw.append(out.argmax(dim=1))

        #objectness fusion
        if cfg.saliency:
            num_classes = out.shape[1]
            saliency_map = batch[3].to(gpu)

            logits_mask = torch.cat(
                    [
                        1 - saliency_map,  # B 1 H W,
                        saliency_map.repeat(1, num_classes - 1, 1, 1)  # B N-1, H, W
                    ],
                    dim=1,
                )  # B N H W
            out_ = (out * logits_mask).softmax(dim=1)
        else:
            out_ = out

        out = out_.argmax(dim=1)
        masks.append(out)
        img_ids += batch[0]

    with open(os.path.join(cfg.output_dir, str(gpu) + '.pkl'), 'wb') as f:
        pickle.dump([torch.cat(masks, dim=0).cpu(), img_ids], f)

    return torch.cat(masks, dim=0).cpu().numpy(), img_ids


def eval_clipconv(cfg):
    log.info(f"Running on {cfg.dataset.split}")
    loader, dataset_raw, num_classes = load_data(cfg, cfg.dataset.split, False, False, cfg.saliency)

    if num_classes == 81:
        class_names = COCO_CLASS_NAMES_ALL
    elif num_classes == 21:
        class_names = VOC_CLASSES

    log.info('Getting predictions')

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(get_predictions, args=(cfg, world_size, class_names), nprocs=world_size, join=True)
    else:
        get_predictions(0, cfg, world_size, class_names)
    log.info('Calculating metrics')
    preds = []
    img_ids = []
    for i in range(world_size):
        with open(os.path.join(cfg.output_dir, str(i) + '.pkl'), 'rb') as f:
            file = pickle.load(f)
            preds.append(file[0])
            img_ids += file[1]

    preds = torch.cat(preds, dim=0).numpy()

    predictions, gt_labels = combine_predictions(preds, img_ids, dataset_raw)
    evaluate_predictions(predictions, gt_labels, n_classes=num_classes, dataset_results_dir=cfg.output_dir,
                               results_file_name=os.path.join(cfg.output_dir, f"results_clip_convv2_detailed.csv"))


def ddp_setup(rank: int, world_size: int, port="12345"):
#    """
# +   Args:
# +       rank: Unique identifier of each process
# +      world_size: Total number of processes
# +   """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port) if type(port) is not str else port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CLIP-DIY evaluation')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)
    eval_clipconv(cfg)

