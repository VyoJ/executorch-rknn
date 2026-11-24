"""MobileViT-XS (apple/mobilevit-x-small) and its preprocessing, for demo D1.

Preprocessing follows the model's preprocessor_config.json: shorter side to
288 (bilinear), centre crop 256, scale to [0, 1], RGB -> BGR. The 160 px
Imagenette images are upscaled.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import MobileViTConfig, MobileViTForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rknn_baseline"))
from prepare import WNID_TO_INDEX, pick  # noqa: E402

WEIGHTS = Path("/opt/data/hf/mobilevit-x-small")
SIZE = 256


class MobileViT(torch.nn.Module):
    """Pixel values in, logits out."""

    def __init__(self):
        super().__init__()
        # transformers 5 loads only safetensors; this checkpoint ships a .bin.
        self.model = MobileViTForImageClassification(MobileViTConfig.from_pretrained(WEIGHTS))
        self.model.load_state_dict(torch.load(WEIGHTS / "pytorch_model.bin", weights_only=True))
        self.eval()

    def forward(self, x):
        return self.model(pixel_values=x).logits


def preprocess(path):
    im = Image.open(path).convert("RGB")
    scale = 288 / min(im.size)
    im = im.resize((round(im.size[0] * scale), round(im.size[1] * scale)), Image.BILINEAR)
    left, top = (im.size[0] - SIZE) // 2, (im.size[1] - SIZE) // 2
    x = np.asarray(im.crop((left, top, left + SIZE, top + SIZE)), np.float32)[:, :, ::-1] / 255.0
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def images(split, per_class):
    """(label, input) pairs from Imagenette."""
    for wnid, path in pick(split, per_class):
        yield WNID_TO_INDEX[wnid], preprocess(path)
