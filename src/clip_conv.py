"""
CLIP-DIY
author: Monika Wysoczanska, Warsaw University of Technology
"""


from typing import Any, List, Optional, Tuple, Union
from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
import torchvision.transforms as T
import torch
import torch.nn.functional as F


_Patches = List[torch.Tensor]


class MetaClipConv(torch.nn.Module):
    def __init__(self, config, class_names, return_intermediate, *args, **kwargs):
        super(MetaClipConv, self).__init__()
        self.return_intermediate = return_intermediate
        self.class_names = class_names
        self._set_model(config)
        self.eval()
        self.to_PIL = T.ToPILImage()
        self.to_t = T.ToTensor()
        self.patch_sizes = config.patch_sizes
        self.strides = config.strides
        self.interpolate_logits = config.interpolate_logits
        self.img_size = config.img_size
        self.align_corners = config.align_corners

        patch_weights = torch.tensor([config.patch_w1, config.patch_w2, config.patch_w3])
        self.register_buffer("patch_weights", patch_weights)

    def _set_model(self, config):
        raise NotImplementedError

    def _extract_patches(
        self,
        images: torch.Tensor,
        *args, **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        extract patches from images
        """
        raise NotImplementedError

    @torch.no_grad()
    def _process_patches(self, patches, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def _group_patches(self, patches, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def infer(self, patches, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _create_prompt(text):
        return 'a photo of an ' + text if text.startswith('aeiou') else 'a photo of a ' + text

    @torch.no_grad()
    def _get_class_embeddings(self, text_model: torch.nn.Module, class_names: List[str]):
        raise NotImplementedError

    def forward(self, images):
        patches = self._extract_patches(images)
        processed_patches = self._process_patches(patches)
        infered_patches, extra_outputs = self.infer(processed_patches)
        return self._group_patches(infered_patches)


class ClipConv(MetaClipConv):
    def __init__(self, config, class_names, return_intermediate=False):
        super(ClipConv, self).__init__(config, class_names, return_intermediate=return_intermediate)
        self.class_names = class_names

    def _set_model(self, config):
        self.image_model = CLIPVisionModelWithProjection.from_pretrained(config.clip_model).eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(config.clip_model)
        self.image_processor = CLIPImageProcessor.from_pretrained(config.clip_model)
        self.logit_scale = CLIPModel.from_pretrained(config.clip_model).logit_scale
        # extract class embeddings

        text_model = CLIPTextModelWithProjection.from_pretrained(config.clip_model).eval()
        self.register_buffer("class_embeddings", self._get_class_embeddings(text_model, self.class_names))

    def clip_conv(self, img, patch_size=32, stride=2):
        B, _, h, w = img.shape
        patches = self._extract_patches(img, patch_size, stride)  # B, 3, npatch, hp, wp  (npatch = (hw // patch_size**2))
        patches = self._process_patches(patches)  # List[PIL.Image]  (B*npatch x (3, hp, wp))

        patches_sims = self.infer(patches)  # B*npatch, C
        num_classes = patches_sims.shape[-1]

        masks = self._group_patches(patches_sims, (B, num_classes, h, w))  # B, C, h, w

        return masks

    def forward(self, images):
        masks = []
        for ps in self.patch_sizes:
            for s in self.strides:
                masks.append(self.clip_conv(images, ps, s))
        mask = torch.mean(torch.stack(masks, dim=0) * self.patch_weights.view((-1,) + masks[0].dim() * (1,)), dim=0)
        if self.return_intermediate:
            return mask, masks
        else:
            return mask

    def _embed_label(self, text_model: torch.nn.Module, label: str) -> torch.Tensor:
        """
        Encode label name into a single vector
        """
        all_prompts = [self._create_prompt(label)]
        l_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
        out = text_model(**l_tokenized).text_embeds
        out = torch.mean(out, dim=0)
        return out

    def _get_class_embeddings(self, text_model: torch.nn.Module, class_names: List[str]):
        aug_embeddings = torch.stack([self._embed_label(text_model, label) for label in class_names])
        # normalize vector
        aug_embeddings = aug_embeddings / aug_embeddings.norm(p=2, dim=-1, keepdim=True)
        return aug_embeddings

    @torch.no_grad()
    def infer(self, b_patches: _Patches):
        """
        infer logits from image patches
        """
        image_processed = self.image_processor(b_patches, return_tensors="pt").to(self.image_model.device)
        image_out = self.image_model(**image_processed)

        # normalized features
        image_embeds = image_out.image_embeds / image_out.image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(self.class_embeddings, image_embeds.t())  #* logit_scale
        logits_per_image = logits_per_text.t()

        return logits_per_image

    def _extract_patches(self, img, patch_size=32, stride=2) -> torch.Tensor:
        patches = img.unfold(2, patch_size, int(patch_size // stride)).unfold(3, patch_size,
                                                                              int(patch_size // stride)).flatten(2, 3)
        return patches

    def _process_patches(self, patches):
        patches = patches.permute(0, 2, 1, 3, 4).flatten(0, 1)  # B, C, npatch, hp, wp -> B*npatches C h w
        return [self.to_PIL(patch) for patch in patches]

    def _group_patches(self, patches, output_shape: tuple) -> torch.Tensor:
        """
        ClipConv patch grouping
        Note: this assumes patches are from a square image
        """
        assert len(output_shape) == 4, "output_shape should be 4D"
        B, C, H, W = output_shape
        patches = patches.reshape(B, -1, C)
        num_patches = patches.shape[1]
        num_patches_w = num_patches_h = int(num_patches**0.5)
        patches = patches.reshape(B, num_patches_h, num_patches_w, C).permute(0, 3, 1, 2)  # B C H W

        if self.interpolate_logits:
            mask = F.interpolate(patches, size=(H, W), mode="bilinear", align_corners=self.align_corners)
        else:
            mask = F.interpolate(patches, size=(H, W), mode="nearest")

        return mask
