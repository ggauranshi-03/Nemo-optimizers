import math
from dataclasses import dataclass
from typing import Optional

import torch
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning.pytorch.optim import OptimizerModule
from torch.optim.optimizer import Optimizer


# Yogi Optimizer Implementation
class Yogi(Optimizer):
    """
    Implements Yogi optimizer.

    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages (default: (0.9, 0.999))
        eps: term added to denominator for numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Yogi, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Yogi does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # FIX: Explicitly move to p.device
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    ).to(p.device)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    ).to(p.device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # Add weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Decay the first moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Yogi update for second moment (key difference from Adam)
                grad_squared = grad.pow(2)
                exp_avg_sq.mul_(beta2).add_(
                    torch.sign(grad_squared - exp_avg_sq) * grad_squared,
                    alpha=1 - beta2,
                )

                # Compute bias-corrected moments
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute step size
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


@dataclass
class YogiConfig:
    lr: float
    betas: tuple
    eps: float
    weight_decay: float


class DistributedOptimizerWrapper:
    """A wrapper to make standard PyTorch optimizers compatible with NeMo's Megatron Strategy."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        # NeMo looks for 'mcore_optimizer' or a direct reference to the torch optimizer
        self.mcore_optimizer = optimizer

    def __getattr__(self, name):
        # Redirect any missing attributes to the underlying optimizer (step, zero_grad, etc.)
        return getattr(self.optimizer, name)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class YogiOptimizerModule(OptimizerModule):
    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        lr_scheduler=None,
    ):
        super().__init__(lr_scheduler=lr_scheduler)
        self.config = YogiConfig(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.lr_scheduler = lr_scheduler

    def optimizers(self, model):
        # 1. Initialize your custom Yogi optimizer
        opt = Yogi(
            model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        # 2. Add the attribute NeMo's MegatronStrategy is looking for.
        # This bypasses the AttributeError without breaking Lightning's type-checks.
        opt.mcore_optimizer = opt

        # 3. Return as a list of valid torch.optim.Optimizer objects
        return [opt]


def get_yogi_optimizer(
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
):
    """
    Create Yogi optimizer module for NeMo.

    Args:
        lr: Learning rate
        betas: Coefficients for computing running averages
        eps: Term for numerical stability
        weight_decay: Weight decay coefficient

    Returns:
        YogiOptimizerModule configured with Yogi optimizer
    """
    return YogiOptimizerModule(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def main():
    """Main function to run pretraining."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NeMo GPT Pretraining with Yogi Optimizer"
    )
    parser.add_argument(
        "--name", type=str, default="gpt_yogi_pretrain", help="Experiment name"
    )
    parser.add_argument("--dir", type=str, default="/results", help="Results directory")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--num_gpus_per_node", type=int, default=1, help="GPUs per node"
    )
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument(
        "--num_attention_heads", type=int, default=12, help="Attention heads"
    )
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Max training steps"
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=32, help="Global batch size"
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=4, help="Micro batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Yogi")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Yogi")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    args = parser.parse_args()

    # Model configuration
    model_config = llm.GPTConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.hidden_size * 4,
        num_attention_heads=args.num_attention_heads,
        seq_length=args.seq_length,
        init_method_std=0.023,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
    )
    tokenizer = AutoTokenizer(pretrained_model_name="gpt2")
    # Create model
    model = llm.GPTModel(config=model_config, tokenizer=tokenizer)

    # Data module
    data = llm.MockDataModule(
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
    )

    # Strategy configuration
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ddp="megatron",
        find_unused_parameters=False,
    )

    # Yogi optimizer
    optimizer = get_yogi_optimizer(
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # Trainer configuration
    # trainer = nl.Trainer(
    #     devices=args.num_gpus_per_node,
    #     num_nodes=args.num_nodes,
    #     max_steps=args.max_steps,
    #     accelerator="gpu",
    #     strategy=strategy,
    #     log_every_n_steps=10,
    #     val_check_interval=100,
    #     limit_val_batches=10,
    #     plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    # )
    # In main(), change your trainer configuration to:
    trainer = nl.Trainer(
        devices=args.num_gpus_per_node,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        # Use standard Lightning precision instead of the nl.MegatronMixedPrecision plugin
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    # Logger configuration
    logger = nl.NeMoLogger(
        name=args.name,
        log_dir=args.dir,
        explicit_log_dir=f"{args.dir}/{args.name}",
        use_datetime_version=False,
        ckpt=nl.ModelCheckpoint(
            save_last=True,
            save_top_k=3,
            every_n_train_steps=100,
            monitor="reduced_train_loss",
            filename="{epoch}-{step}-{reduced_train_loss:.4f}",
        ),
    )

    # Resume configuration
    resume = nl.AutoResume(resume_if_exists=False)

    # Train the model
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        optim=optimizer,
        resume=resume,
    )


if __name__ == "__main__":
    main()

