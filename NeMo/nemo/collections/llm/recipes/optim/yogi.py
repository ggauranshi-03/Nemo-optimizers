import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Callable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import nemo_run as run
from nemo.lightning import LightningModule
# Reference Yogi optimizer: https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/yogi.py
# NeMo & Megatron Imports
from megatron.core.optimizer import (
    OptimizerConfig,
    DistributedOptimizer,
    Float16OptimizerWithFloat16Params,
    ConstantGradScaler,
    DynamicGradScaler,
    _get_param_groups_and_buffers
)
# from nemo.collections.nlp.parts.megatron_optimizer import (
#     _get_param_groups_and_buffers, 
# )
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule

# -------------------------------------------------------------------------
# 1. THE YOGI OPTIMIZER IMPLEMENTATION
# -------------------------------------------------------------------------
class Yogi(Optimizer):
    """
    Implements Yogi Optimizer Algorithm.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
        initial_accumulator: float = 1e-6,
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            initial_accumulator=initial_accumulator,
            weight_decay=weight_decay,
        )
        super(Yogi, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Yogi does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = nn.init.constant_(
                        torch.empty_like(p.data, memory_format=torch.preserve_format),
                        group["initial_accumulator"],
                    )
                    state["exp_avg_sq"] = nn.init.constant_(
                        torch.empty_like(p.data, memory_format=torch.preserve_format),
                        group["initial_accumulator"],
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_squared = grad.mul(grad)

                # Yogi Update: v_t = v_{t-1} + sign(g^2 - v_{t-1}) * g^2
                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_squared),
                    grad_squared,
                    value=-(1 - beta2),
                )

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# -------------------------------------------------------------------------
# 2. CONFIGURATION CLASS
# -------------------------------------------------------------------------
@dataclass
class YogiOptimizerConfig(OptimizerConfig):
    # We override the name, though our custom module won't check this string logic
    optimizer: str = "yogi"
    initial_accumulator: float = 1e-6
    yogi_beta1: float = 0.9
    yogi_beta2: float = 0.999
    yogi_eps: float = 1e-3


# -------------------------------------------------------------------------
# 3. CUSTOM MEGATRON MODULE (The Logic Replacement)
# -------------------------------------------------------------------------
class YogiMegatronOptimizerModule(MegatronOptimizerModule):
    """
    Custom module that forces the use of the Yogi optimizer class defined above,
    bypassing NeMo's string-based registry lookup.
    """
    def configure_optimizers(self, model: LightningModule):
        # We assume the model is a Megatron model with `model_chunks` (standard in NeMo LLMs)
        # If not, fall back to self.model.parameters()
        model_chunks = getattr(model, "model_chunks", [model])
        
        # 1. Get Parameter Groups using standard NeMo utility
        # This separates bias/layernorm (no weight decay) from other weights
        param_groups, buffers = _get_param_groups_and_buffers(
            model_chunks,
            model_chunk_offset=0,
            config=self.config,
            config_overrides=None,
            filter_fn=lambda g: True,
            buffer_name='buffers',
        )

        # 2. Instantiate Yogi manually
        optimizer = Yogi(
            param_groups,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(self.config.yogi_beta1, self.config.yogi_beta2),
            eps=self.config.yogi_eps,
            initial_accumulator=self.config.initial_accumulator,
        )

        # 3. Setup Gradient Scaler (Standard Megatron Logic)
        grad_scaler = None
        if self.config.fp16 or self.config.bf16 or self.config.use_distributed_optimizer:
            if self.config.loss_scale:
                grad_scaler = ConstantGradScaler(self.config.loss_scale)
            elif self.config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=self.config.initial_loss_scale,
                    min_scale=self.config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=self.config.loss_scale_window,
                    hysteresis=self.config.hysteresis,
                )

        # 4. Helper function to initialize Yogi state (required for DistributedOptimizer to allocate buffers)
        def init_state_fn(opt):
            for group in opt.param_groups:
                for p in group['params']:
                    if len(opt.state[p]) == 0:
                        opt.state[p]['step'] = 0
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data) + self.config.initial_accumulator

        # 5. Wrap in DistributedOptimizer (Sharding) or Float16Optimizer
        if self.config.use_distributed_optimizer:
            # ZeRO-1: Shard optimizer state across GPUs
            optimizer = DistributedOptimizer(
                optimizer,
                self.config,
                grad_scaler,
                init_state_fn,
                model_chunks=model_chunks,
                per_model_buffers=buffers,
                # These groups are usually handled automatically by NeMo/Megatron context,
                # but explicit passing is sometimes required if not using defaults.
                # Assuming standard NeMo setup, defaults inside DistributedOptimizer often work 
                # or NeMo sets up `parallel_state` correctly.
            )
        else:
            # Standard Mixed Precision Wrapper (No sharding)
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer,
                self.config,
                grad_scaler,
                init_state_fn,
            )

        # 6. Setup Scheduler
        if self.lr_scheduler:
            scheduler = self.lr_scheduler.create_scheduler(optimizer, max_steps=self.trainer.max_steps)
            return [optimizer], [scheduler]
        
        return [optimizer]


# -------------------------------------------------------------------------
# 4. THE FACTORY FUNCTION (Use this in your CLI/Recipe)
# -------------------------------------------------------------------------
@run.cli.factory
def distributed_fused_yogi_with_cosine_annealing(
    precision: str = "bf16-mixed",
    warmup_steps: int = 2000,
    constant_steps: int = 0,
    yogi_beta1: float = 0.9,
    yogi_beta2: float = 0.999,
    yogi_eps: float = 1e-3,
    initial_accumulator: float = 1e-6,
    max_lr: float = 1e-2,
    min_lr: Optional[float] = None,
    clip_grad: float = 1.0,
    weight_decay: float = 0.0,
    use_distributed_optimizer: bool = True,
) -> run.Config[MegatronOptimizerModule]:
    
    # 1. Config Object
    opt_cfg = run.Config(
        YogiOptimizerConfig,
        optimizer="yogi", # This string is ignored by our custom module, but kept for logging
        lr=max_lr,
        weight_decay=weight_decay,
        bf16=precision == "bf16-mixed",
        fp16=precision == "16-mixed",
        yogi_beta1=yogi_beta1,
        yogi_beta2=yogi_beta2,
        yogi_eps=yogi_eps,
        initial_accumulator=initial_accumulator,
        use_distributed_optimizer=use_distributed_optimizer,
        clip_grad=clip_grad,
    )

    # 2. Scheduler Config
    min_lr = min_lr if min_lr is not None else (0.1 * max_lr)
    sched = run.Config(
        CosineAnnealingScheduler,
        warmup_steps=warmup_steps,
        constant_steps=constant_steps,
        min_lr=min_lr,
    )

    # 3. Return Custom Module Config
    # Note: We return YogiMegatronOptimizerModule, NOT the standard MegatronOptimizerModule
    return run.Config(
        YogiMegatronOptimizerModule,
        config=opt_cfg,
        lr_scheduler=sched,
    )