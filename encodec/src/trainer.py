import torch

from typing import Dict, List, Union, Any

from torch import nn
from transformers import Trainer
from .balancer import LossBalancer


class EncodecTrainer(Trainer):
    def __init__(self, *args, num_losses, use_balancer=True, beta=0.999, R=1.0, **kwargs):
        """
        Initialize the BalancedTrainer with additional parameters for LossBalancer.

        Args:
            num_losses (int): Number of different loss terms to balance.
            use_balancer (bool): Whether to use the LossBalancer.
            beta (float): Decay rate for moving average of gradient norms.
            R (float): Reference norm for balancing.
        """
        super().__init__(*args, **kwargs)
        self.use_balancer = use_balancer
        if self.use_balancer:
            self.loss_balancer = LossBalancer(num_losses=num_losses, beta=beta, R=R)

    def compute_generator_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss using the LossBalancer.

        Args:
            model (torch.nn.Module): The model being trained.
            inputs (dict): The inputs to the model.
            return_outputs (bool): Whether to return the model outputs.

        Returns:
            torch.Tensor: The balanced loss.
            dict (optional): The model outputs if return_outputs is True.
        """
        outputs = model(**inputs)

        loss_values = outputs.losses
        model_output = outputs.model_output

        if not self.use_balancer:
            return (loss_values, outputs) if return_outputs else loss_values

        loss_weights = model.config.loss_weights
        balanced_loss = self.loss_balancer.apply_balancer(loss_values, loss_weights, model_output)
        return (balanced_loss, outputs) if return_outputs else balanced_loss

    def compute_discriminator_loss(self, model, inputs, return_outputs=False):
        fake_features = model.encodec(inputs['input_values']).audio_codes
        real_features = inputs['audio_codes']



    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor: