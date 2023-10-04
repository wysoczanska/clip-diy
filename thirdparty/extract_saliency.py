# Modified version by Monika Wysoczanska
# __________________________________
# Copyright 2022 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from FOUND.model import FoundModel
from misc import load_config
from torchvision import transforms as T
TH = 0.5
NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Evaluation of FOUND',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--img-dir", type=str, default="data/COCO", help="Image path."
    )
    parser.add_argument(
        "--model-weights", type=str, default="FOUND/data/weights/decoder_weights.pt",
    )
    parser.add_argument(
        "--config", type=str, default="FOUND/configs/found_DUTS-TR.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
    )
    parser.add_argument(
        "--drop-probs", action='store_true')

    args = parser.parse_args()

    # Saving dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Configuration
    config = load_config(args.config)

    # ------------------------------------
    # Load the model
    model = FoundModel(vit_model=config.model["pre_training"],
                        vit_arch=config.model["arch"],
                        vit_patch_size=config.model["patch_size"],
                        enc_type_feats=config.found["feats"],
                        bkg_type_feats=config.found["feats"],
                        bkg_th=config.found["bkg_th"])
    # Load weights
    model.decoder_load_weights(args.model_weights)
    model.eval()
    print(f"Model {args.model_weights} loaded correctly.")

    # Load the image
    import os

    img_paths = os.listdir(args.img_dir)
    for i_path in img_paths:
        with open(os.path.join(args.img_dir, i_path), "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

            t = T.Compose([T.ToTensor(), NORMALIZE])
            img_t = t(img)[None,:,:,:]
            inputs = img_t.to("cuda")

        # Forward step
        with torch.no_grad():
            preds, _, shape_f, att = model.forward_step(inputs, for_eval=True)

        # Apply FOUND
        sigmoid = nn.Sigmoid()
        h, w = img_t.shape[-2:]
        preds_up = F.interpolate(
            preds, scale_factor=model.vit_patch_size, mode="bilinear", align_corners=False
        )[..., :h, :w]

        if args.drop_probs:
            preds_up = (
                (sigmoid(preds_up.detach())).squeeze(0).float()
            )
        else:
            preds_up = (
                (sigmoid(preds_up.detach()) > TH).squeeze(0).float()
            )

        with open(os.path.join(args.output_dir, f"{i_path.split('.')[0]}.npy"), 'wb') as f:
            np.save(f, preds_up.cpu().squeeze().numpy())




