# -*- coding: utf-8 -*-
# *** indent: 4 spaces ***
"""Utility functions for the diffusion models tutorial."""

import os
from argparse import Namespace

import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader


def plot_images(images: torch.Tensor) -> None:
    """
    Plot a grid of images using matplotlib.

    Parameters
    ----------
    images : torch.Tensor
        A tensor containing the images to be plotted. The tensor is expected to be in the format
        (N, C, H, W) where N is the number of images, C is the number of channels, H is the height,
        and W is the width.
    """
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images: torch.Tensor, path: str, **kwargs: dict) -> None:
    """
    Save a grid of images to a specified path.

    Parameters
    ----------
    images : torch.Tensor
        A tensor containing the images to be saved. The tensor is expected to be in the format
        (N, C, H, W) where N is the number of images, C is the number of channels, H is the height,
        and W is the width.
    path : str
        The file path where the image grid will be saved.
    **kwargs : dict
        Additional keyword arguments to pass to `torchvision.utils.make_grid`.
    """
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args: Namespace) -> DataLoader:
    """
    Create a DataLoader for the dataset specified in the arguments.

    Parameters
    ----------
    args : Namespace
        A namespace object containing the dataset path, image size, and batch size.

    Returns
    -------
    DataLoader
        A DataLoader object for the dataset.
    """
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name: str) -> None:
    """
    Set up directories for logging models and results.

    Parameters
    ----------
    run_name : str
        The name of the run for which to set up logging directories.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
