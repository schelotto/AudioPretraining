import torch
from typing import Dict, List
from transformers import Trainer
from collections import defaultdict


class LossBalancer:
    def __init__(self, num_losses: int, beta: float = 0.999, R: float = 1.0, eps: float = 1e-12):
        """
        Initializes the LossBalancer.

        Args:
            num_losses (int): The number of different losses to balance.
            beta (float): The decay factor for the exponential moving average.
            R (float): The reference norm for balancing.
        """
        self.num_losses = num_losses
        self.R = R
        self.beta = beta
        self.eps = eps
        self.ema_grad_norms = defaultdict(float)
        self.fix = defaultdict(float)

    def _update_ema(self, grad_norms: Dict[str, float]):
        """
        Updates the exponential moving average (EMA) for the gradient norms.

        Args:
            grad_norms (dict): The current norms of the gradients.

        Returns:
            dict: The updated EMA norms.
        """
        for key, value in grad_norms.items():
            self.ema_grad_norms[key] = self.ema_grad_norms[key] * self.beta + value
            self.fix[key] = self.fix[key] * self.beta + 1.0

        return {key: self.ema_grad_norms[key] / self.fix[key] for key in grad_norms.keys()}

    def compute_balancer_weights(self, grads: List[torch.Tensor], loss_weights: List[float]):
        """
        Computes the balancing weights for each loss term.

        Args:
            grads (list of torch.Tensor): The gradients of each loss term.
            loss_weights (list of float): The initial weights for each loss term (λᵢ).

        Returns:
            list of torch.Tensor: The adjusted gradients.
        """
        # Compute current norms of gradients
        current_grad_norms = {f'grad_{i}': torch.norm(grad, p=2).item() for i, grad in enumerate(grads)}

        # Update EMA norms
        ema_grad_norms = self._update_ema(current_grad_norms)

        # Compute adjusted gradients
        adjusted_grads = []
        sum_weights = sum(loss_weights)

        for i, grad in enumerate(grads):
            norm_factor = ema_grad_norms[f'grad_{i}'] + self.eps
            adjusted_weight = (self.R * loss_weights[i] / sum_weights) / norm_factor
            adjusted_grads.append(adjusted_weight * grad)

        return adjusted_grads

    def apply_balancer(self, loss_values: List[torch.Tensor], loss_weights: List[float], model_output: torch.Tensor):
        """
        Balances the losses and applies the balancer to compute the final loss.

        Args:
            loss_values (list of torch.Tensor): The individual loss values.
            loss_weights (list of float): The initial weights for each loss term (λᵢ).
            model_output (torch.Tensor): The output of the model (x̂).

        Returns:
            torch.Tensor: The final balanced loss.
        """
        # Compute the gradients for each loss with respect to model output
        grads = [torch.autograd.grad(loss, model_output, retain_graph=True)[0] for loss in loss_values]

        adjusted_grads = self.compute_balancer_weights(grads, loss_weights)
        final_grad = sum(adjusted_grads)

        model_output.backward(final_grad)
        final_loss = sum(loss_values)
        return final_loss


class BalancedTrainer(Trainer):
    def __init__(self, *args, num_losses, beta=0.999, R=1.0, **kwargs):
        """
        Initialize the BalancedTrainer with additional parameters for LossBalancer.

        Args:
            num_losses (int): Number of different loss terms to balance.
            beta (float): Decay rate for moving average of gradient norms.
            R (float): Reference norm for balancing.
        """
        super().__init__(*args, **kwargs)
        self.loss_balancer = LossBalancer(num_losses=num_losses, beta=beta, R=R)

    def compute_loss(self, model, inputs, return_outputs=False):
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

        loss_weights = model.config.loss_weights
        balanced_loss = self.loss_balancer.apply_balancer(loss_values, loss_weights, model_output)
        return (balanced_loss, outputs) if return_outputs else balanced_loss
