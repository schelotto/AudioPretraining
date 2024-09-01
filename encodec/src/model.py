import torch

from transformers.models.encodec import EncodecModel
from loss import EncodecLoss, EncodecLossConfig
from mssftd import MultiScaleSTFTDiscriminator
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union


class EncodecPretrainingOutput(ModelOutput):
    """
    Base class for outputs of Encodec model pretraining.

    Args:
        loss (torch.Tensor, optional): The pretraining loss, if computed.
        audio_values (torch.Tensor): The reconstructed audio values.
        audio_codes (torch.Tensor): The discrete audio codes.
    """
    loss: Optional[torch.Tensor] = None
    audio_values: Optional[torch.Tensor] = None
    audio_codes: Optional[torch.Tensor] = None


class EncodecModelForPretraining(EncodecModel):
    def __init__(self, config: EncodecLossConfig):
        super().__init__(config)
        self.loss_fn = EncodecLoss(config)
        self.discriminator = MultiScaleSTFTDiscriminator(
            filters=config.num_filters,
            in_channels=config.audio_channels
        )

        for param in self.discriminator.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_values: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            bandwidth: Optional[float] = None,
            audio_codes: Optional[torch.Tensor] = None,
            audio_scales: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            return_loss: Optional[bool] = True,
            target_values: Optional[torch.Tensor] = None,
    ) -> Union[EncodecPretrainingOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform the forward pass, optionally computing the loss.

        Args:
            input_values (torch.Tensor): Input audio tensor (shape: [batch_size, channels, seq_length]).
            padding_mask (torch.Tensor, optional): Mask to indicate padding.
            bandwidth (float, optional): Target bandwidth for encoding. Defaults to None.
            audio_codes (torch.Tensor, optional): Precomputed audio codes. Defaults to None.
            audio_scales (torch.Tensor, optional): Scales corresponding to the audio codes. Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary or a tuple. Defaults to None.
            return_loss (bool, optional): Whether to compute and return the loss. Defaults to True.
            target_values (torch.Tensor, optional): Target audio tensor for reconstruction (shape: [batch_size, channels, seq_length]).

        Returns:
            EncodecPretrainingOutput: Contains loss (if computed), audio values, and audio codes.
        """
        # Get the encoded codes and decoded audio
        output = super().forward(
            input_values=input_values,
            padding_mask=padding_mask,
            bandwidth=bandwidth,
            audio_codes=audio_codes,
            audio_scales=audio_scales,
            return_dict=True
        )
        x_hat = output.audio_values

        loss = None
        # If return_loss is True and target_values is provided, compute the loss
        if return_loss and target_values is not None:
            discriminator_outputs, fake_features = self.discriminator(x_hat)
            real_features = [real_feature.detach() for real_feature in self.discriminator(target_values)[1]]

            z, zq = self.encoder(input_values), self.quantizer.encode(self.encoder(input_values))
            loss = self.loss_fn(
                target_values, x_hat, z, zq,
                discriminator_outputs, real_features, fake_features
            )

        if return_dict:
            return EncodecPretrainingOutput(
                loss=loss,
                audio_values=x_hat,
                audio_codes=output.audio_codes,
            )
        else:
            return (loss, x_hat) if loss is not None else (x_hat, output.audio_codes)
