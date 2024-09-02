import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from transformers.models.encodec import EncodecConfig


class EncodecLossConfig(EncodecConfig):
    def __init__(
            self,
            loss_weights=None,
            alpha_l1=1.0,
            alpha_l2=1.0,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            **kwargs,
    ):
        super().__init__(**kwargs)
        if loss_weights is None:
            loss_weights = [0.1, 1.0, 3.0, 3.0, 1.0]

        assert len(loss_weights) == 5, "The loss_weights list must have exactly 5 elements."

        self.loss_weights = loss_weights
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        base_config = EncodecConfig.from_pretrained(pretrained_model_name_or_path)

        # Loss-specific defaults
        loss_specific_kwargs = {
            "loss_weights": kwargs.pop("loss_weights", [0.1, 1.0, 3.0, 3.0, 1.0]),
            "alpha_l1": kwargs.pop("alpha_l1", 1.0),
            "alpha_l2": kwargs.pop("alpha_l2", 1.0),
            "n_fft": kwargs.pop("n_fft", 1024),
            "win_length": kwargs.pop("win_length", 1024),
            "hop_length": kwargs.pop("hop_length", 256),
            "n_mels": kwargs.pop("n_mels", 80),
        }

        config_dict = base_config.to_dict()
        config_dict.update(loss_specific_kwargs)
        return cls(**config_dict)

    def to_encodec_config(self):
        encodec_dict = {key: value for key, value in self.to_dict().items() if
                        key not in ['loss_weights', 'alpha_l1', 'alpha_l2']}
        return EncodecConfig(**encodec_dict)


class EncodecLoss(nn.Module):
    def __init__(self, config: EncodecLossConfig):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        # Weights from the config
        self.loss_weights = config.loss_weights
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
        # Reconstruction loss in time domain
        loss_t = self.l1_loss(x, x_hat)

        # Reconstruction loss in frequency domain across different scales
        loss_f = 0.0
        for mel_spectrogram in self.mel_spectrograms:
            mel_x = mel_spectrogram(x)
            mel_x_hat = mel_spectrogram(x_hat)
            loss_f_l1 = self.l1_loss(mel_x, mel_x_hat)
            loss_f_l2 = self.l2_loss(mel_x, mel_x_hat)
            loss_f += self.alpha_l1 * loss_f_l1 + self.alpha_l2 * loss_f_l2

        loss_f /= len(self.mel_spectrograms)

        # Adversarial loss
        loss_g = 0.0
        K = len(discriminator_outputs)
        for output in discriminator_outputs:
            loss_g += torch.mean(torch.clamp(1.0 - output, min=0.0))
        loss_g /= K

        # Feature matching loss
        feat_loss = 0.0
        for real_feature, fake_feature in zip(real_features, fake_features):
            for real_feat, fake_feat in zip(real_feature, fake_feature):
                feat_loss += self.l1_loss(fake_feat, real_feat)

        feat_loss /= K

        # Commitment loss
        loss_w = self.l2_loss(z.detach(), zq)
        return loss_t, loss_f, loss_g, feat_loss, loss_w
