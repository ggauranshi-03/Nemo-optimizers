import math
import argparse
import os
import torch
from torch.optim.optimizer import Optimizer
from nemo.lightning.pytorch.optim import OptimizerModule

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zero-power / orthogonalization.
    """
    assert G.ndim == 2, f"Expected 2D tensor, got {G.ndim}D"
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    if G.size(0) > G.size(1):
        G = G.t()
        transposed = True
    else:
        transposed = False
    
    norm = G.norm() + eps
    X = G / norm
    X = X.bfloat16()
    
    for _ in range(steps):
        A = X.t() @ X
        B = b * A + c * A @ A
        X = X @ (a * torch.eye(X.size(1), device=X.device, dtype=X.dtype) + B)
    
    if transposed:
        X = X.t()
    
    return X.float()

# ============================================================================ #
#                            Hybrid Muon Optimizer                             #
# ============================================================================ #

class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,               
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_w_lr: float = 0.003,       
        adam_w_betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov, 
            ns_steps=ns_steps,
            adam_w_lr=adam_w_lr,
            adam_w_betas=adam_w_betas,
            weight_decay=weight_decay,
            eps=eps
        )
        super().__init__(params, defaults)
        self.log_interval = 10  # Print stats every 10 steps

    def _classify_param(self, p):
        # Embeddings: large first dimension
        is_embedding = (p.ndim == 2 and p.size(0) > 10000)
        # Norms and biases: 1D
        is_norm_or_bias = (p.ndim < 2)
        # Linear weights: 2D but not embeddings
        is_linear_weight = (p.ndim == 2 and not is_embedding)
        return is_linear_weight

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Counters for verification
        muon_updates = 0
        adam_updates = 0
        skipped = 0

        for group in self.param_groups:
            # Muon params
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']
            
            # AdamW params
            adam_lr = group['adam_w_lr']
            beta1, beta2 = group['adam_w_betas']
            eps = group['eps']

            for p in group['params']:
                # --- MEGATRON GRADIENT CHECK ---
                grad = p.grad
                if grad is None and hasattr(p, 'main_grad'):
                    grad = p.main_grad

                if grad is None:
                    skipped += 1
                    continue
                
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['use_muon'] = self._classify_param(p)
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                use_muon = state['use_muon']

                if use_muon:
                    # ================= MUON UPDATE ================= #
                    muon_updates += 1
                    
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    
                    if nesterov:
                        g = buf.clone().add_(grad, alpha=1 - momentum)
                    else:
                        g = buf
                    
                    # Run Newton-Schulz
                    g_ortho = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    
                    rows, cols = g.size()
                    scale = max(1, rows / cols) ** 0.5
                    g_ortho *= scale
                    
                    p.add_(g_ortho, alpha=-lr)

                else:
                    # ================= ADAMW UPDATE ================= #
                    adam_updates += 1
                    
                    if weight_decay != 0:
                        p.mul_(1 - adam_lr * weight_decay)
                    
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    step_size = adam_lr / bias_correction1
                    bias_correction2_sqrt = math.sqrt(bias_correction2)
                    
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        step_count = 0
        if len(self.param_groups) > 0 and len(self.param_groups[0]['params']) > 0:
             p0 = self.param_groups[0]['params'][0]
             if p0 in self.state:
                 step_count = self.state[p0]['step']

        if step_count % self.log_interval == 0 or step_count == 1:
            print(f"\n[OPTIMIZER CHECK step {step_count}]")
            print(f"  > Muon Updates (Matrices): {muon_updates}")
            print(f"  > Adam Updates (Vectors):  {adam_updates}")
            print(f"  > Skipped (No Grad):       {skipped}")
            if muon_updates > 0:
                print("  [VERIFIED] Newton-Schulz Orthogonalization is ACTIVE.")
            else:
                print("  [WARNING] No Muon updates occurred!")

        return loss

# ============================================================================ #
#                           Muon Module Wrapper                                #
# ============================================================================ #
class MuonOptimizerModule(OptimizerModule):
    def __init__(self, lr: float, adam_w_lr: float, weight_decay: float, lr_scheduler=None):
        super().__init__(lr_scheduler=lr_scheduler)
        self.config = None 
        self.lr = lr
        self.adam_w_lr = adam_w_lr
        self.weight_decay = weight_decay

    def optimizers(self, model):
        # Filter for parameters that actually require gradients
        params = [p for p in model.parameters() if p.requires_grad]
        return [Muon(
            params,
            lr=self.lr,
            adam_w_lr=self.adam_w_lr,
            weight_decay=self.weight_decay,
            momentum=0.95,
            nesterov=True,
            ns_steps=5
        )]
