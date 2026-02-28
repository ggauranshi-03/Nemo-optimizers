import math
import argparse
import os
import torch
from torch.optim.optimizer import Optimizer
from nemo.lightning.pytorch.optim import OptimizerModule
class Yogi(Optimizer):
    """Yogi optimizer"""
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Yogi, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"]

            for p in group["params"]:
                # --- 1. ACCESS GRADIENT (MEGATRON COMPATIBLE) ---
                grad = p.grad if p.grad is not None else getattr(p, 'main_grad', None)
                if grad is None:
                    continue

                state = self.state[p]

                # --- 2. INITIALIZE STATE (ONLY ONCE) ---
                # We use float32 for states to maintain precision
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                # --- 3. APPLY WEIGHT DECAY ---
                # We use a temporary variable 'g' to avoid modifying the original grad buffer
                if wd != 0:
                    g = grad.add(p, alpha=wd)
                else:
                    g = grad

                # --- 4. UPDATE MOMENTS (IN-PLACE TO SAVE MEMORY) ---
                # exp_avg = beta1 * exp_avg + (1 - beta1) * g
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)

                # Yogi Update Logic:
                # v = v - (1-beta2) * sign(v - g^2) * g^2
                # We compute g^2 once to save memory
                g2 = g.pow(2)
                
                # We need sign(exp_avg_sq - g2). We compute this carefully.
                # To save memory, we don't store the whole 'diff' tensor.
                exp_avg_sq.addcmul_(
                    (exp_avg_sq - g2).sign_(), 
                    g2, 
                    value=-(1 - beta2)
                )

                # --- 5. COMPUTE DENOMINATOR & UPDATE WEIGHTS ---
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                
                # step_size = lr / bias_correction1
                # denom = sqrt(exp_avg_sq / bias_correction2) + eps
                
                curr_lr = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Final update: p = p - curr_lr * (exp_avg / denom)
                # addcdiv_ is highly memory efficient for this
                p.addcdiv_(exp_avg, denom, value=-curr_lr)

                # Cleanup large temporary tensors for the next parameter
                if wd != 0: del g
                del g2

        return loss
# ============================================================================ #
#                           Yogi Module Wrapper                                #
# ============================================================================ #
class YogiOptimizerModule(OptimizerModule):
    """
    Wraps the custom Yogi optimizer to make it compatible with NeMo's
    OptimizerModule interface.
    """
    def __init__(self, lr: float, betas: tuple, weight_decay: float, lr_scheduler=None):
        super().__init__(lr_scheduler=lr_scheduler)

        self.config = None 
        
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

    def optimizers(self, model):
        """
        This method is called by NeMo to instantiate the actual optimizer.
        """
        return [Yogi(
            model.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay
        )]