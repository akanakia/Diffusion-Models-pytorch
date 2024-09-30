# -*- coding: utf-8 -*-
# *** indent: 4 spaces ***
"""Modules for the diffusion models tutorial."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    """
    Exponential Moving Average (EMA) class for updating model parameters.

    Parameters
    ----------
    beta : float
        The decay rate for the moving average.
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        """
        Update the moving average model parameters.

        Parameters
        ----------
        ma_model : nn.Module
            The model with moving average parameters.
        current_model : nn.Module
            The current model with the latest parameters.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """
        Compute the updated average of the parameters.

        Parameters
        ----------
        old : torch.Tensor
            The old parameter values.
        new : torch.Tensor
            The new parameter values.

        Returns
        -------
        torch.Tensor
            The updated parameter values.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 2000) -> None:
        """
        Perform a step of the EMA update.

        Parameters
        ----------
        ema_model : nn.Module
            The model with moving average parameters.
        model : nn.Module
            The current model with the latest parameters.
        step_start_ema : int, optional
            The step at which to start updating the EMA, by default 2000.
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model: nn.Module, model: nn.Module) -> None:
        """
        Reset the EMA model parameters to the current model parameters.

        Parameters
        ----------
        ema_model : nn.Module
            The model with moving average parameters.
        model : nn.Module
            The current model with the latest parameters.
        """
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    """
    Self-Attention layer for processing input tensors.

    Parameters
    ----------
    channels : int
        Number of input channels.
    size : int
        Spatial size of the input tensor.
    """

    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Self-Attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after applying self-attention and feed-forward layers.
        """
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Double convolutional layer with optional residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int, optional
        Number of intermediate channels, by default None.
    residual : bool, optional
        Whether to use a residual connection, by default False.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, residual: bool = False) -> None:
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DoubleConv layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after applying double convolution.
        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class DownSampler(nn.Module):
    """
    DownSampler module for reducing the spatial dimensions of the input tensor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    emb_dim : int, optional
        Dimensionality of the embedding, by default 256.
    """

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).
        t : torch.Tensor
            Time step tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the UNet model.
        """
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpSampler(nn.Module):
    """
    UpSampler module for increasing the spatial dimensions of the input tensor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    emb_dim : int, optional
        Dimensionality of the embedding, by default 256.
    """

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Up module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).
        skip_x : torch.Tensor
            Skip connection tensor from the corresponding Down module.
        t : torch.Tensor
            Time step tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the Up module.
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    """
    UNet model for image generation.

    Parameters
    ----------
    c_in : int, optional
        Number of input channels, by default 3.
    c_out : int, optional
        Number of output channels, by default 3.
    time_dim : int, optional
        Dimensionality of the time embedding, by default 256.
    device : str, optional
        Device to run the model on, by default "cuda".
    """

    def __init__(self, c_in: int = 3, c_out: int = 3, time_dim: int = 256, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = DownSampler(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = DownSampler(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = DownSampler(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = UpSampler(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = UpSampler(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = UpSampler(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        """
        Compute the positional encoding for the given time steps.

        Parameters
        ----------
        t : torch.Tensor
            Time step tensor.
        channels : int
            Number of channels for the positional encoding.

        Returns
        -------
        torch.Tensor
            Positional encoding tensor.
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).
        t : torch.Tensor
            Time step tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the UNet model.
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    """
    UNet model with conditional generation capability.

    Parameters
    ----------
    c_in : int, optional
        Number of input channels, by default 3.
    c_out : int, optional
        Number of output channels, by default 3.
    time_dim : int, optional
        Dimensionality of the time embedding, by default 256.
    num_classes : int, optional
        Number of classes for conditional generation, by default None.
    device : str, optional
        Device to run the model on, by default "cuda".
    """

    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = DownSampler(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = DownSampler(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = DownSampler(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = UpSampler(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = UpSampler(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = UpSampler(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        """
        Compute the positional encoding for the given time steps.

        Parameters
        ----------
        t : torch.Tensor
            Time step tensor.
        channels : int
            Number of channels for the positional encoding.

        Returns
        -------
        torch.Tensor
            Positional encoding tensor.
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet_conditional model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).
        t : torch.Tensor
            Time step tensor.
        y : torch.Tensor
            Label tensor for conditional generation.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the UNet_conditional model.
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == "__main__":
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
