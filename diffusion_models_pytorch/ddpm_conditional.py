# -*- coding: utf-8 -*-
# *** indent: 4 spaces ***
"""Conditional diffusion models for image generation tutorial."""
import copy
import logging
import os
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .modules import EMA, UNet_conditional
from .utils import get_data, plot_images, save_images, setup_logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    """A sample diffusion model for image generation."""

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        """Prepare the noise schedule for the diffusion process."""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to images according to the diffusion process.

        Parameters
        ----------
        x : torch.Tensor
            The input images.
        t : torch.Tensor
            The timesteps for the diffusion process.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the noised images and the noise added.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n: int) -> torch.Tensor:
        """
        Sample random timesteps for the diffusion process.

        Parameters
        ----------
        n : int
            The number of timesteps to sample.

        Returns
        -------
        torch.Tensor
            A tensor containing the sampled timesteps.
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model: nn.Module, n: int, labels: torch.Tensor, cfg_scale: float = 3) -> torch.Tensor:
        """
        Generate samples from the diffusion model.

        Parameters
        ----------
        model : nn.Module
            The neural network model used for sampling.
        n : int
            The number of samples to generate.
        labels : torch.Tensor
            The labels for conditional generation.
        cfg_scale : float, optional
            The scale for classifier-free guidance (default is 3).

        Returns
        -------
        torch.Tensor
            A tensor containing the generated samples.
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args: Namespace) -> None:
    """
    Train the diffusion model.

    Parameters
    ----------
    args : Namespace
        The arguments for training, including hyperparameters and configurations.

    """
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    data_size = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * data_size + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, "optim.pt"))


def _launch():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 14
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == "__main__":
    _launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
