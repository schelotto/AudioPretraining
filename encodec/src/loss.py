import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from transformers.models.encodec import EncodecConfig


class EncodecLossConfig(EncodecConfig):
    def __init__(
        self,
        lambda_t=1.0,
        lambda_f=1.0,
        lambda_g=1.0,
        lambda_feat=1.0,
        lambda_w=1.0,
        alpha_l1=1.0,
        alpha_l2=1.0,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_t = lambda_t
        self.lambda_f = lambda_f
        self.lambda_g = lambda_g
        self.lambda_feat = lambda_feat
        self.lambda_w = lambda_w
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels


class EncodecLoss(nn.Module):
    def __init__(self, config: EncodecLossConfig):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        # Weights from the config
        self.lambda_t = config.lambda_t
        self.lambda_f = config.lambda_f
        self.lambda_g = config.lambda_g
        self.lambda_feat = config.lambda_feat
        self.lambda_w = config.lambda_w
        self.alpha_l1 = config.alpha_l1
        self.alpha_l2 = config.alpha_l2

        # Mel-spectrogram transformation
        self.scales = [2 ** i for i in range(5, 12)]
        self.mel_spectrograms = [
            transforms.MelSpectrogram(
                sample_rate=config.sampling_rate,
                n_fft=scale,
                win_length=scale,
                hop_length=scale // 4,
                n_mels=config.n_mels
            ) for scale in self.scales
        ]

    def forward(self, x, x_hat, z, zq, discriminator_outputs, real_features, fake_features):
        # reconstruction loss in time domain
        loss_t = self.l1_loss(x, x_hat)

        # reconstruction loss in frequency domain across different scales
        loss_f = 0.0
        for mel_spectrogram in self.mel_spectrograms:
            mel_x = mel_spectrogram(x)
            mel_x_hat = mel_spectrogram(x_hat)
            loss_f_l1 = self.l1_loss(mel_x, mel_x_hat)
            loss_f_l2 = self.l2_loss(mel_x, mel_x_hat)
            loss_f += self.alpha_l1 * loss_f_l1 + self.alpha_l2 * loss_f_l2

        loss_f /= len(self.mel_spectrograms)
        reconstruction_loss = self.lambda_t * loss_t + self.lambda_f * loss_f

        # adversarial loss
        K = len(discriminator_outputs)
        loss_g = 0.0
        for output in discriminator_outputs:
            loss_g += torch.mean(torch.clamp(1.0 - output, min=0.0))
        loss_g /= K
        adv_loss = self.lambda_g * loss_g

        # feature matching loss
        feat_loss = 0.0
        for real_feats, fake_feats in zip(real_features, fake_features):
            diff = [torch.abs(real_feat - fake_feat) for real_feat, fake_feat in zip(real_feats, fake_feats)]
            diff = torch.stack(diff)

            mean_real_feats = [torch.mean(torch.abs(real_feat), dim=[1, 2], keepdim=True) for real_feat in real_feats]
            mean_real_feats = torch.stack(mean_real_feats)

            normalized_diff = diff / (mean_real_feats + 1e-8)
            feat_loss += normalized_diff.mean()

        feat_loss /= K
        feat_loss = self.lambda_feat * feat_loss

        # commitment loss
        loss_w = self.l2_loss(z.detach(), zq)
        commitment_loss = self.lambda_w * loss_w

        loss = reconstruction_loss + adv_loss + feat_loss + commitment_loss
        return loss

