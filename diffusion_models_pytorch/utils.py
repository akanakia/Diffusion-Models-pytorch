# -*- coding: utf-8 -*-
# *** indent: 4 spaces ***
"""Utility functions for the diffusion models tutorial."""

import os
from io import BytesIO
from typing import Any

import PIL
import torch
from datasets import Dataset, DatasetDict, Image, load_dataset
from matplotlib import pyplot as plt
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as ttv2
from torchvision.utils import make_grid


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


def save_images(images: torch.Tensor, save_path: str, **kwargs: dict) -> None:
    """
    Save a grid of images to a specified path.

    Parameters
    ----------
    images : torch.Tensor
        A tensor containing the images to be saved. The tensor is expected to be in the format
        (N, C, H, W) where N is the number of images, C is the number of channels, H is the height,
        and W is the width.
    save_path : str
        The file path where the image grid will be saved.
    **kwargs : dict
        Additional keyword arguments to pass to `torchvision.utils.make_grid`.
    """
    grid = make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(save_path)


def get_data(dataset_name: str, img_col: str, img_resize: int, split: str = "train", **kwargs) -> DataLoader:
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
    transforms = ttv2.Compose(
        [
            ttv2.RandomResizedCrop(img_resize, scale=(0.8, 1.0)),
            # ttv2.ToDtype(torch.float32, scale=True),
            ttv2.ToTensor(),
            ttv2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    set_hf_cache_dir()
    ds = load_hf_dataset(dataset_name, img_col=img_col, split=split)
    dataloader = DataLoader(ds, collate_fn=ImageTransformCollator(transforms, img_col), **kwargs)

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


def set_hf_cache_dir(cache_dir: str = "D:/datasets") -> None:
    """
    Set the Hugging Face cache directory.

    Parameters
    ----------
    cache_dir : str
        The directory to use as the cache for Hugging Face datasets, by default "D:/datasets".
    """
    os.environ["HF_DATASETS_CACHE"] = cache_dir


def load_hf_dataset(dataset_name: str, img_col: str, split: str = "train") -> Dataset | DatasetDict:
    """
    Load a dataset from the Hugging Face Hub.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load.
    split : str, optional
        The specific split of the dataset to load (e.g., 'train', 'test'). If None, the entire dataset is loaded.

    Returns
    -------
    Dataset or DatasetDict
        The loaded dataset or a specific split of the dataset.
    """

    def _has_valid_image(row):
        try:
            PIL.Image.open(BytesIO(row[img_col]["bytes"]))
        except UnidentifiedImageError:
            return False
        return True

    ds = load_dataset(dataset_name, cache_dir=os.environ["HF_DATASETS_CACHE"])[split]
    ds = ds.cast_column(img_col, Image(decode=False))
    ds = ds.filter(_has_valid_image)
    ds = ds.cast_column(img_col, Image(decode=True))
    return ds


class ImageTransformCollator:
    """
    A collator for image datasets that applies transformations to the images.

    Parameters
    ----------
    transforms : Callable
        The transformations to apply to the images.
    """

    def __init__(self, transforms: callable, img_col: str) -> None:
        self.transforms = transforms
        self.img_col = img_col

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Apply the transformations to the images in the batch.

        Parameters
        ----------
        batch : dict
            A dictionary containing the images and labels.

        Returns
        -------
        dict
            A dictionary containing the transformed images and labels.
        """
        X = list()
        y = list()
        for row in batch:
            X.append(self.transforms(row[self.img_col]))
            y.append(row["label"])
        return dict(X=torch.stack(X), y=torch.IntTensor(y))
