"""Load SB3 checkpoints across file-like-reader differences in PyTorch builds."""

import io
import zipfile
from unittest.mock import patch

import torch
from stable_baselines3 import PPO


def load_ppo(path, **kwargs):
    """Buffer inner ZIP members for Torch's reader, retaining weights_only=True.

    The installed Torch 2.13 CPU build reads BytesIO correctly but fails on a
    ZipExtFile's serialization ID. SB3 passes those members directly to Torch.
    This narrow, temporary adapter leaves checkpoints and installed libraries
    untouched. Model loading in this application is single-threaded.
    """
    original_load = torch.load

    def buffered_load(source, *args, **options):
        if isinstance(source, zipfile.ZipExtFile):
            source = io.BytesIO(source.read())
        return original_load(source, *args, **options)

    with patch("torch.load", buffered_load):
        return PPO.load(path, **kwargs)
