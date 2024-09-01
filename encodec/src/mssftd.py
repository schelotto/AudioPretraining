import typing as tp
import torch
import torchaudio

from torch import nn
from torch.nn.utils import spectral_norm, weight_norm
from einops import rearrange
from typing import List, Tuple


CONV_NORMALIZATIONS = frozenset([
    'none', 'weight_norm', 'spectral_norm',
    'time_layer_norm', 'layer_norm', 'time_group_norm'
])


LogitsType = torch.Tensor
FeatureMapType = List[torch.Tensor]
DiscriminatorOutput = Tuple[List[LogitsType], List[FeatureMapType]]


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimension
    before running the normalization and moves them back afterward.
    """
    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = rearrange(x, 'b t ... -> b ... t')
        return x


def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """Returns the appropriate normalization module."""
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
    def __init__(self, *args, norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2
    )


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator."""
    def __init__(
        self, filters: int, in_channels: int = 1, out_channels: int = 1,
        n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
        max_filters: int = 1024, filters_scale: int = 1, kernel_size: tp.Tuple[int, int] = (3, 9),
        dilations: tp.List[int] = [1, 2, 4], stride: tp.Tuple[int, int] = (1, 2),
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

        self.convs.append(NormConv2d(
            in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm
        ))

        self.conv_post = NormConv2d(
            out_chs, out_channels, kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm
        )

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        for layer in self.convs:
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator."""
    def __init__(
        self, filters: int, in_channels: int = 1, out_channels: int = 1,
        n_ffts: tp.List[int] = [1024, 2048, 512], hop_lengths: tp.List[int] = [256, 512, 128],
        win_lengths: tp.List[int] = [1024, 2048, 512], **kwargs
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
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
