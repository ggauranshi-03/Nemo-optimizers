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

class AdaMuon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,              
        betas: tuple = (0.9, 0.95),      
        ns_steps: int = 5,
        adam_w_lr: float = 0.003,       
        adam_w_betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, 
            betas=betas,                
            ns_steps=ns_steps,
            adam_w_lr=adam_w_lr,
            adam_w_betas=adam_w_betas,
            weight_decay=weight_decay,
            eps=eps
        )
        super().__init__(params, defaults)
        self.log_interval = 10 

    def _classify_param(self, p):
        # Same classification logic as before
        is_embedding = (p.ndim == 2 and p.size(0) > 10000)
        is_norm_or_bias = (p.ndim < 2)
        is_linear_weight = (p.ndim == 2 and not is_embedding)
        return is_linear_weight

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_updates = 0
        adam_updates = 0
        skipped = 0

        for group in self.param_groups:
            # Common params
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            # Muon params
            muon_beta1, muon_beta2 = group['betas']
            ns_steps = group['ns_steps']
            
            # AdamW params
            adam_lr = group['adam_w_lr']
            adam_beta1, adam_beta2 = group['adam_w_betas']
            eps = group['eps']

            for p in group['params']:
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
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                use_muon = state['use_muon']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step_t = state['step']

                if use_muon:

                    muon_updates += 1
                    
                    # 1. Update Momentum (M_t)
                    # M_t = beta * M_{t-1} + (1-beta) * G_t
                    exp_avg.mul_(muon_beta1).add_(grad, alpha=1 - muon_beta1)
                    
                    # 2. Orthogonalize Momentum (O_t)
                    # Note: Algorithm runs NS on M_t directly
                    M_t = exp_avg
                    O_t = zeropower_via_newtonschulz5(M_t, steps=ns_steps)
                    
                    # 3. Update Second Moment (v_t) using Orthogonalized Direction
                    # v_t = beta2 * v_{t-1} + (1-beta2) * O_t^2  <--- CRITICAL CHANGE
                    # Algorithm uses element-wise squaring of O_t
                    exp_avg_sq.mul_(muon_beta2).addcmul_(O_t, O_t, value=1 - muon_beta2)
                    
                    # 4. Adaptive Update (O_hat)
                    # v_hat = v_t / (1 - beta2^t)
                    # o_hat = O_t / (sqrt(v_hat) + eps)
                    bias_correction2 = 1 - muon_beta2 ** step_t
                    v_hat = exp_avg_sq / bias_correction2
                    denom = v_hat.sqrt().add_(eps)
                    O_hat = O_t / denom
                    
                    # 5. RMS-aligned Rescaling
                    # scaling_factor = 0.2 / (RMS(O_hat) + eps)
                    # RMS = sqrt(mean(square(x)))
                    rms = O_hat.pow(2).mean().sqrt()
                    scaling_factor = 0.2 / (rms + eps)
                    
                    # 6. Apply Update with Weight Decay
                    # W_{t+1} = W_t - lr * (scaling_factor * O_hat + lambda * W_t)
                    
                    # Calculate the full update term: (scale * O_hat + wd * W)
                    update_term = O_hat.mul_(scaling_factor)
                    if weight_decay != 0:
                        update_term.add_(p, alpha=weight_decay)
                        
                    p.add_(update_term, alpha=-lr)

                else:
                    # ================================================================= #
                    #                  Standard AdamW (Auxiliary)                      #
                    # ================================================================= #
                    adam_updates += 1
                    
                    # Standard Weight Decay (Decoupled)
                    if weight_decay != 0:
                        p.mul_(1 - adam_lr * weight_decay)

                    # Update Moments
                    exp_avg.mul_(adam_beta1).add_(grad, alpha=1 - adam_beta1)
                    exp_avg_sq.mul_(adam_beta2).addcmul_(grad, grad, value=1 - adam_beta2)
                    
                    bias_correction1 = 1 - adam_beta1 ** step_t
                    bias_correction2 = 1 - adam_beta2 ** step_t
                    
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
            print(f"  > AdaMuon Updates (Strict Algo 1): {muon_updates}")
            print(f"  > AdamW Updates (Auxiliary):       {adam_updates}")

        return loss

# ============================================================================ #
#                           Muon Module Wrapper                                #
# ============================================================================ #
class AdaMuonOptimizerModule(OptimizerModule):
    def __init__(self, lr: float, adam_w_lr: float, weight_decay: float, lr_scheduler=None):
        super().__init__(lr_scheduler=lr_scheduler)
        self.config = None 
        self.lr = lr
        self.adam_w_lr = adam_w_lr
        self.weight_decay = weight_decay

    def optimizers(self, model):
        # Filter for parameters that actually require gradients
        params = [p for p in model.parameters() if p.requires_grad]
        return [AdaMuon(         
            params,
            lr=self.lr,
            adam_w_lr=self.adam_w_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),   
            ns_steps=5
        )]