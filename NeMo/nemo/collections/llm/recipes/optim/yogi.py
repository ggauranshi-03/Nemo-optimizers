import torch
import nemo_run as run
from torch.optim import Optimizer
from typing import Optional

# --- 1. Define the Optimizer Class directly here ---
class Yogi(Optimizer):
    """Implementation of Yogi: Adaptive Backward Propagation for Training Deep Neural Networks."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Yogi, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Yogi Update logic
                grad_sq = grad.pow(2)
                sign = torch.sign(grad_sq - exp_avg_sq)
                exp_avg_sq.addcmul_(sign, grad_sq, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2**0.5) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

# --- 2. Define the Recipe Helper ---
def distributed_yogi_with_cosine_annealing(
    precision: str = "bf16-mixed",
    warmup_steps: int = 500,
    constant_steps: int = 0,
    min_lr: float = 1e-5,
    max_lr: float = 1e-4,
    clip_grad: float = 1.0,
):
    """
    Recipe helper that uses the locally defined Yogi class.
    """
    from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
    
    # We leverage the existing Adam recipe structure but 
    # force it to use our local Yogi class as the optimizer.
    recipe = distributed_fused_adam_with_cosine_annealing(
        precision=precision,
        warmup_steps=warmup_steps,
        constant_steps=constant_steps,
        min_lr=min_lr,
        max_lr=max_lr,
        clip_grad=clip_grad,
    )
    
    # Override the optimizer class in the configuration
    recipe.optimizer_type = run.Config(Yogi)
    
    return recipe