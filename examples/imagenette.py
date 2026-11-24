"""Imagenette images for calibration and evaluation, labelled with their
ImageNet-1k class index. The aot image unpacks Imagenette at /opt/data."""

from pathlib import Path

import torch
from torchvision.transforms import v2 as T

IMAGENETTE = Path("/opt/data/imagenette2-160")

# Imagenette WordNet ids -> ImageNet-1k class index.
WNID_TO_INDEX = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

crop = T.Compose([T.Resize(256), T.CenterCrop(224)])
normalize = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def pick(split, per_class):
    """The first per_class images of each class in split ("train" or "val"), as (wnid, path)."""
    for wnid in sorted(WNID_TO_INDEX):
        for path in sorted((IMAGENETTE / split / wnid).glob("*.JPEG"))[:per_class]:
            yield wnid, path
