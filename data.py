"""CIFAR-10 images and contiguous time windows from a complex CSI trace."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from models import ChannelPredictor


ROOT = Path(__file__).resolve().parent


def load_images(data_dir, train, limit=None):
    images = CIFAR10(root=data_dir, train=train, download=True, transform=ToTensor())
    if limit is None:
        return images
    if not 1 <= limit <= len(images):
        raise ValueError(f"Image count must be between 1 and {len(images)}")
    return Subset(images, range(limit))


def load_csi(path):
    trace = np.load(path, allow_pickle=False, mmap_mode="r")
    if trace.ndim != 3 or trace.shape[1:] != (2, 2) or not np.iscomplexobj(trace):
        raise ValueError("CSI must be a complex NumPy array of shape (time, 2, 2)")
    return torch.from_numpy(np.array(trace, dtype=np.complex64, order="C", copy=True))


def csi_windows(trace, start, count, slots):
    """Build one observed history per four-slot forecast and one actual link per image."""
    history = ChannelPredictor.history
    first = start * slots
    stop = first + history + count * slots
    if stop > len(trace):
        raise ValueError(f"Need {stop} CSI samples for this batch, found {len(trace)}")
    refresh = ChannelPredictor.refresh
    windows = trace[first:stop].unfold(0, history + refresh, refresh).movedim(-1, 1)
    actual = trace[first + history:stop].reshape(count, slots, 2, 2)
    return windows[:, :history], actual


def load_weights(model, checkpoint):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=True)
    return model
