import torch
from torch.optim import Optimizer

class Yogi(Optimizer):
    """Implementation of Yogi: Adaptive Backward Propagation for Training Deep Neural Networks.
    Yogi is an improvement over Adam that controls the adaptive learning rate better.
    """
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
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first moment
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Yogi second moment update:
                # v_t = v_{t-1} + (1-beta2) * sign(g^2 - v_{t-1}) * g^2
                grad_sq = grad.pow(2)
                sign = torch.sign(grad_sq - exp_avg_sq)
                exp_avg_sq.addcmul_(sign, grad_sq, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2**0.5) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss