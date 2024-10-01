# -*- coding: utf-8 -*-
# *** indent: 4 spaces ***
"""Train a diffusion model for image generation."""
import argparse
from datetime import datetime

import yaml
from munch import munchify

from diffusion_models_pytorch.ddpm import train_ddpm
from diffusion_models_pytorch.ddpm_conditional import train_ddpm_conditional
from diffusion_models_pytorch.utils import setup_output_paths


def _load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Model training yaml configuration path.")
    parser.add_argument("-r", "--run_id", type=str, default=None, help="Run ID for the model training.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.run_id is None:
        args.run_id = f"{config['model']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config["run_id"] = args.run_id
    return munchify(config)


def _launch():
    config = _load_config()
    config.result_path, config.trained_path, config.logs_path = setup_output_paths(config.output_path, config.run_id)
    if config.model == "ddpm":
        train_ddpm(config)
    elif config.model == "ddpm_conditional":
        train_ddpm_conditional(config)
    else:
        raise ValueError(f"Unknown model: {config.model}")


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
