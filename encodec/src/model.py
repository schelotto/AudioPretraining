import torch
from transformers.models.encodec import EncodecPreTrainedModel, EncodecModel, EncodecConfig
from .loss import EncodecLoss, EncodecLossConfig
from .mssftd import MultiScaleSTFTDiscriminator
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from encodec import EncodecModel


@dataclass
class EncodecPretrainingOutput(ModelOutput):
    """
    Output class for Encodec model pretraining.

    Attributes:
        audio_values (torch.Tensor, optional): Reconstructed audio waveform.
        audio_codes (torch.Tensor, optional): Discrete audio codes generated by the encoder.
        losses (List[torch.Tensor], optional): List of individual loss terms.
    """

    audio_values: Optional[torch.Tensor] = None
    audio_codes: Optional[torch.Tensor] = None
    losses: Optional[List[torch.Tensor]] = None


class EncodecModelForPretraining(EncodecPreTrainedModel):
    def __init__(self, config: EncodecLossConfig):
        super().__init__(config)
        self.config = config
        self.encodec_config = config.to_encodec_config()

        self.encodec = EncodecModel(self.encodec_config)
        self.loss_fn = EncodecLoss(config)
        self.discriminator = MultiScaleSTFTDiscriminator(
            filters=config.num_filters,
            in_channels=config.audio_channels
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load the pretraining model from a pretrained EncodecConfig with additional loss configuration.

        Args:
            pretrained_model_name_or_path (str or os.PathLike): Path to the pretrained model or model identifier from HuggingFace model hub.
            kwargs: Additional configuration parameters for the EncodecLossConfig.

        Returns:
            EncodecModelForPretraining: An instance of the pretraining model with the loaded configuration.
        """
        encodec_config = EncodecConfig.from_pretrained(pretrained_model_name_or_path)
        config = EncodecLossConfig(
            **encodec_config.to_dict(),  # Start with the base config
            **kwargs  # Override with any additional loss-specific arguments
        )
        return cls(config)

    def forward(
            self,
            input_values: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            bandwidth: Optional[float] = None,
            audio_codes: Optional[torch.Tensor] = None,
            audio_scales: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = True,
            return_loss: Optional[bool] = True,
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

        Returns:
            EncodecPretrainingOutput: Contains loss (if computed), audio values, and audio codes.
        """
        # Get the encoded codes and decoded audio
        output = self.encodec(
            input_values=input_values,
            padding_mask=padding_mask,
            bandwidth=bandwidth,
            audio_codes=audio_codes,
            audio_scales=audio_scales,
            return_dict=True
        )
        x_hat = output.audio_values

        if return_loss:
            z = self.encodec.encoder(input_values)
            zq = self.encodec.quantizer.encode(z)

            discriminator_outputs, fake_features = self.discriminator(x_hat)
            _, real_features = self.discriminator(input_values)

            loss_t, loss_f, loss_g, feat_loss, loss_w = self.loss_fn(
                input_values, x_hat, z, zq,
                discriminator_outputs, real_features, fake_features
            )

            losses = [loss_t, loss_f, loss_g, feat_loss, loss_w]
            if return_dict:
                return EncodecPretrainingOutput(
                    losses=losses,
                    audio_values=x_hat,
                    audio_codes=output.audio_codes,
                )
            else:
                return (losses, x_hat) if losses is not None else (x_hat, output.audio_codes)
        return output
