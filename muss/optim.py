"""Optimizer and learning-rate schedule used by the released fine-tuned models.

The Adam update is adapted from fairseq (MIT; see docs/licenses/fairseq_MIT.txt).
Its epsilon is applied before the second-moment bias correction.
"""

import math
import torch


class MUSSAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.98), eps=1e-6):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse or parameter.dtype != torch.float32:
                    raise ValueError("MUSSAdam requires dense float32 parameters and gradients")
                state = self.state[parameter]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
                state["step"] += 1
                mean, variance = state["exp_avg"], state["exp_avg_sq"]
                mean.mul_(beta1).add_(grad, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denominator = variance.sqrt().add_(group["eps"])
                step_size = group["lr"] * math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])
                parameter.addcdiv_(mean, denominator, value=-step_size)
        return loss


class WarmupInverseSqrt(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup=100):
        if warmup < 1:
            raise ValueError("warmup must be positive")
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self):
        updates = self.last_epoch
        if updates < self.warmup:
            return [updates * (lr / self.warmup) for lr in self.base_lrs]
        return [(lr * self.warmup ** 0.5) * updates ** -0.5 for lr in self.base_lrs]
