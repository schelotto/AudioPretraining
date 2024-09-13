import torch
import torchaudio

from torch import nn
from torch.nn.utils import spectral_norm, weight_norm
from einops import rearrange
from typing import List, Tuple, Dict, Any, Union


CONV_NORMALIZATIONS = frozenset([
    'none', 'weight_norm', 'spectral_norm',
    'time_layer_norm', 'layer_norm', 'time_group_norm'
])


LogitsType = torch.Tensor
FeatureMapType = List[torch.Tensor]
DiscriminatorOutput = Tuple[List[LogitsType], List[FeatureMapType]]


class ConvLayerNorm(nn.LayerNorm):
    """
    Custom LayerNorm that supports convolutional layers.

    Args:
        normalized_shape (Union[int, List[int], torch.Size]): Shape of the input to be normalized.

    Forward:
        x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width, time].

    Returns:
        torch.Tensor: Normalized tensor with the same shape as input.
    """

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        # Rearrange the tensor for layer normalization
        x = rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        # Revert the tensor to its original shape
        x = rearrange(x, 'b t ... -> b ... t')
        return x


def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    """
    Apply the specified normalization to a given module.

    Args:
        module (nn.Module): Module to which the normalization is applied.
        norm (str): Type of normalization to apply. Must be one of the values in CONV_NORMALIZATIONS.

    Returns:
        nn.Module: Module with the specified normalization applied.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """
    Get the normalization module based on the type of normalization specified.

    Args:
        module (nn.Module): The module to be normalized.
        causal (bool): Whether the normalization should be causal. Default is False.
        norm (str): Type of normalization. Default is 'none'.
        norm_kwargs (dict): Additional keyword arguments for the normalization.

    Returns:
        nn.Module: The normalization module or identity if none is specified.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    return nn.Identity()


class NormConv2d(nn.Module):
    """
    2D Convolutional layer with integrated normalization.

    Args:
        args: Positional arguments for nn.Conv2d.
        norm (str): Type of normalization to apply. Default is 'none'.
        norm_kwargs (dict): Additional keyword arguments for the normalization.
        kwargs: Additional keyword arguments for nn.Conv2d.

    Forward:
        x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

    Returns:
        torch.Tensor: Convolved and normalized tensor.
    """

    def __init__(self, *args, norm: str = 'none', norm_kwargs: Dict[str, Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


def get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    """
    Calculate 2D padding for convolutional layers.

    Args:
        kernel_size (Tuple[int, int]): Size of the convolutional kernel.
        dilation (Tuple[int, int]): Dilation of the convolutional kernel. Default is (1, 1).

    Returns:
        Tuple[int, int]: Padding values for height and width.
    """
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2
    )


class DiscriminatorSTFT(nn.Module):
    """
    Single-scale STFT Discriminator for audio data.

    Args:
        filters (int): Number of filters in the convolutional layers.
        in_channels (int): Number of input channels. Default is 1.
        out_channels (int): Number of output channels. Default is 1.
        n_fft (int): Size of FFT for STFT. Default is 1024.
        hop_length (int): Length of hop between STFT windows. Default is 256.
        win_length (int): Window size for STFT. Default is 1024.
        max_filters (int): Maximum number of filters in the network. Default is 1024.
        filters_scale (int): Scale factor for filters. Default is 1.
        kernel_size (Tuple[int, int]): Kernel size for Conv2d. Default is (3, 9).
        dilations (List[int]): List of dilation values for Conv2d layers. Default is [1, 2, 4].
        stride (Tuple[int, int]): Stride for Conv2d. Default is (1, 2).
        normalized (bool): Whether to normalize STFT magnitudes. Default is True.
        norm (str): Type of normalization to apply. Default is 'weight_norm'.
        activation (str): Activation function to use. Default is 'LeakyReLU'.
        activation_params (dict): Parameters for the activation function.

    Forward:
        x (torch.Tensor): Input tensor of shape [batch_size, channels, sequence_length].

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: Logits and feature maps from each layer.
    """

    def __init__(
            self, filters: int, in_channels: int = 1, out_channels: int = 1,
            n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
            max_filters: int = 1024, filters_scale: int = 1, kernel_size: Tuple[int, int] = (3, 9),
            dilations: List[int] = [1, 2, 4], stride: Tuple[int, int] = (1, 2),
            normalized: bool = True, norm: str = 'weight_norm',
            activation: str = 'LeakyReLU', activation_params: dict = {'negative_slope': 0.2}
    ):
        super().__init__()
        self.activation = getattr(nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window_fn=torch.hann_window, normalized=normalized, center=False, pad_mode=None, power=None
        )
        spec_channels = 2 * in_channels
        self.convs = nn.ModuleList()

        # First convolution layer
        self.convs.append(NormConv2d(
            spec_channels, filters, kernel_size=kernel_size,
            padding=get_2d_padding(kernel_size), norm=norm
        ))

        in_chs = min(filters_scale * filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * filters, max_filters)
            self.convs.append(NormConv2d(
                in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)), norm=norm
            ))
            in_chs = out_chs

        # Final convolution layers
        self.convs.append(NormConv2d(
            in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm
        ))

        self.conv_post = NormConv2d(
            out_chs, out_channels, kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the STFT Discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, sequence_length].

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Logits and feature maps from each layer.
        """
        fmap = []
        z = self.spec_transform(x)  # STFT transform [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)  # Concatenate real and imaginary parts
        z = rearrange(z, 'b c w t -> b c t w')  # Rearrange to shape [B, channels, time, width]
        for layer in self.convs:
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-scale STFT Discriminator.

    Args:
        filters (int): Number of filters in the convolutional layers.
        in_channels (int): Number of input channels. Default is 1.
        out_channels (int): Number of output channels. Default is 1.
        n_ffts (List[int]): List of FFT sizes for each scale.
        hop_lengths (List[int]): List of hop lengths for each scale.
        win_lengths (List[int]): List of window lengths for each scale.
        kwargs: Additional keyword arguments passed to DiscriminatorSTFT.

    Forward:
        x (torch.Tensor): Input tensor of shape [batch_size, channels, sequence_length].

    Returns:
        DiscriminatorOutput: Tuple containing lists of logits and feature maps from each scale.
    """

    def __init__(
            self, filters: int, in_channels: int = 1, out_channels: int = 1,
            n_ffts: List[int] = [1024, 2048, 512], hop_lengths: List[int] = [256, 512, 128],
            win_lengths: List[int] = [1024, 2048, 512], **kwargs
    ):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(
                filters, in_channels=in_channels, out_channels=out_channels,
                n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs
            ) for i in range(len(n_ffts))
        ])

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        """
        Forward pass through the multi-scale STFT Discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, sequence_length].

        Returns:
            DiscriminatorOutput: Tuple containing lists of logits and feature maps from each scale.
        """
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
