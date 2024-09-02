import torch
from transformers import Trainer


class LossBalancer:
    def __init__(self, num_losses, beta: float = 0.999, R: float = 1.0):
        """
        Initialize the LossBalancer with the number of losses, decay rate, and reference norm.

        Args:
            num_losses (int): Number of different loss terms to balance.
            beta (float): Decay rate for moving average of gradient norms.
            R (float): Reference norm for balancing.
        """
        self.num_losses = num_losses
        self.beta = beta
        self.R = R
        self.moving_avg_grad_norms = [0.0] * num_losses

    def update_moving_avg(self, grads):
        """
        Update the moving average of gradient norms.

        Args:
            grads (list of torch.Tensor): List of gradients for each loss term.
        """
        for i, grad in enumerate(grads):
            grad_norm = torch.norm(grad, p=2).item()
            self.moving_avg_grad_norms[i] = self.beta * self.moving_avg_grad_norms[i] + (1 - self.beta) * grad_norm

    def compute_balancer_weights(self, grads, loss_weights):
        """
        Compute the balancing weights for each gradient.

        Args:
            grads (list of torch.Tensor): List of gradients for each loss term.
            loss_weights (list of float): List of weights for each loss term.

        Returns:
            list of torch.Tensor: List of adjusted gradients.
        """
        sum_weights = sum(loss_weights)
        return [(self.R * loss_weights[i] / sum_weights) / (self.moving_avg_grad_norms[i] + 1e-8) * grad for i, grad in enumerate(grads)]

    def apply_balancer(self, loss_values, loss_weights, model_output):
        """
        Apply the balancer to the loss values and compute the backward pass.

        Args:
            loss_values (list of torch.Tensor): List of loss values.
            loss_weights (list of float): List of weights for each loss term.
            model_output (torch.Tensor): Model output tensor.

        Returns:
            torch.Tensor: Sum of loss values.
        """
        grads = [torch.autograd.grad(loss, model_output, retain_graph=True)[0] for loss in loss_values]

        self.update_moving_avg(grads)
        adjusted_grads = self.compute_balancer_weights(grads, loss_weights)

        model_output.backward(sum(adjusted_grads))
        return sum(loss_values)


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
