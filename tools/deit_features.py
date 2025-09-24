"""Feature extractor wrappers for DeiT backbones (timm)."""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# code borrowed from: https://github.com/facebookresearch/deit

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models import create_model
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from settings import dropout_rate

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


def get_pretrained_weights_path(model_name: str) -> str:
    """Return official DeiT weight URL for selected backbone."""

    finetune = ''
    if model_name in ["deit_small_patch16_224", "deit_base_patch16_224", "deit_tiny_patch16_224",
                      "deit_tiny_distilled_patch16_224"]:
        if model_name == "deit_small_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
        elif model_name == "deit_base_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
        elif model_name == "deit_tiny_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
        elif model_name == "deit_tiny_distilled_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'
    return finetune


def get_pretrained_weights(model_name: str, model: nn.Module) -> nn.Module:
    """Load DeiT weights and interpolate positional embeddings if needed."""

    finetune = get_pretrained_weights_path(model_name)
    if finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(finetune, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    # interpolate position embedding if image size differs
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)
    return model



class DistilledVisionTransformer(VisionTransformer):
    """VisionTransformer variant with a distillation token head."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):

        # not sure if we would employ a teacher model in 
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x#, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch_features(pretrained=False, **kwargs):
    """Return DeiT tiny backbone without classification head."""
    base_arch = 'deit_tiny_patch16_224'
    model = create_model(
    base_arch,
    pretrained=False,
    num_classes=200,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = get_pretrained_weights(base_arch, model)
        del model.head
    return model

@register_model
def deit_small_patch_features(pretrained=False, **kwargs):
    """Return DeiT small backbone without classification head."""
    base_arch = 'deit_small_patch16_224'
    model = create_model(
    base_arch,
    pretrained=False,
    num_classes=200,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = get_pretrained_weights(base_arch, model)
        del model.head
    return model
