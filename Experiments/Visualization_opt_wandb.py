import math
import argparse
from dataclasses import dataclass
import torch

# NeMo imports
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning.pytorch.optim import OptimizerModule
from torch.optim.optimizer import Optimizer
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule

# WandB Import
from pytorch_lightning.loggers import WandbLogger

# --- 1. OPTIMIZER DEFINITIONS (Same as before) ---
class Yogi(Optimizer):
    """(Standard Yogi Implementation)"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Yogi, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad
                if grad.is_sparse: raise RuntimeError("Yogi does not support sparse gradients")
                
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format).to(p.device)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format).to(p.device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Yogi specific update
                grad_squared = grad.pow(2)
                exp_avg_sq.mul_(beta2).add_(
                    torch.sign(grad_squared - exp_avg_sq) * grad_squared,
                    alpha=1 - beta2,
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

@dataclass
class OptimizerConfig:
    lr: float
    betas: tuple
    eps: float
    weight_decay: float
    optimizer_name: str

class CustomOptimizerModule(OptimizerModule):
    def __init__(self, config: OptimizerConfig, lr_scheduler=None):
        super().__init__(lr_scheduler=lr_scheduler)
        self.config = config

    def optimizers(self, model):
        if self.config.optimizer_name == "yogi":
            opt = Yogi(model.parameters(), lr=self.config.lr, betas=self.config.betas, 
                       eps=self.config.eps, weight_decay=self.config.weight_decay)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=self.config.lr, betas=self.config.betas, 
                                    eps=self.config.eps, weight_decay=self.config.weight_decay)
        opt.mcore_optimizer = opt
        return [opt]

# --- 2. MAIN TRAINING LOOP WITH WANDB ---

def main():
    parser = argparse.ArgumentParser(description="NeMo GPT Training with WandB")
    parser.add_argument("--optimizer", type=str, default="yogi", choices=["yogi", "adamw"])
    parser.add_argument("--project_name", type=str, default="nemo-optimizer-benchmark", help="WandB Project Name")
    parser.add_argument("--run_name", type=str, default=None, help="Name for this specific run")
    parser.add_argument("--data_prefix", type=str, default="my_real_data", help="Prefix of the .bin/.idx files")
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    # 1. Setup WandB Logger
    # If run_name is not provided, we create one: e.g., "yogi-run"
    run_name = args.run_name if args.run_name else f"{args.optimizer}-run"
    
    wandb_logger = WandbLogger(
        name=run_name,
        project=args.project_name,
        log_model=False,  # Set to True if you want to upload checkpoints to cloud
    )

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer(pretrained_model_name="gpt2")
    model_config = llm.GPTConfig(
        num_layers=4,
        hidden_size=512,
        ffn_hidden_size=2048,
        num_attention_heads=8,
        seq_length=1024,
        init_method_std=0.02,
        make_vocab_size_divisible_by=128,
    )
    model = llm.GPTModel(config=model_config, tokenizer=tokenizer)

    # 3. Data
    data = PreTrainingDataModule(
        paths=[f"{args.data_prefix}_text_document"],
        seq_length=1024,
        global_batch_size=8,
        micro_batch_size=2,
        tokenizer=tokenizer,
        num_workers=0
    )

    # 4. Strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ddp="megatron",
    )

    # 5. Optimizer
    opt_config = OptimizerConfig(
        lr=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        optimizer_name=args.optimizer
    )
    optimizer = CustomOptimizerModule(config=opt_config)

    # 6. NeMo Logger (Attaching WandB here)
    nemo_logger = nl.NeMoLogger(
        name=run_name,
        log_dir="results",
        wandb=wandb_logger,  # <--- CRITICAL: Attach the WandB logger here
        use_datetime_version=False,
    )

    # 7. Trainer
    trainer = nl.Trainer(
        devices=1,
        accelerator="gpu",
        strategy=strategy,
        max_steps=args.max_steps,
        precision="bf16-mixed",
        log_every_n_steps=5, # Logs to WandB every 5 steps
        enable_checkpointing=False,
    )

    print(f"Starting training with {args.optimizer.upper()} | Logging to WandB project: {args.project_name}")
    
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        optim=optimizer,
        log=nemo_logger,  # Pass the logger wrapper
    )

if __name__ == "__main__":
    main()