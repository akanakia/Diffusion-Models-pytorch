# -*- coding: utf-8 -*-
# *** indent: 4 spaces ***
"""Train a diffusion model for image generation."""
import argparse
from datetime import datetime

from diffusion_models_pytorch.ddpm import train_ddpm
from diffusion_models_pytorch.ddpm_conditional import train_ddpm_conditional


def _parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.model = "ddpm"  # or "ddpm_conditional"
    args.run_id = f"{args.model}_{datetime.now().strftime("%Y%m%d%H%M%S")}"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset = "amaye15/landscapes"  # or "uoft-cs/cifar10"
    args.img_col = "pixel_values"  # or "img"
    args.device = "cuda"
    args.lr = 3e-4
    return args


def _launch():
    args = _parse_args()
    if args.model == "ddpm":
        train_ddpm(args)
    elif args.model == "ddpm_conditional":
        train_ddpm_conditional(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    _launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
