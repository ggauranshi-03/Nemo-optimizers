import math
import argparse
from dataclasses import dataclass
import torch
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning.pytorch.optim import OptimizerModule
from torch.optim.optimizer import Optimizer
# Import the real data module
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule

# --- 1. OPTIMIZER DEFINITIONS ---

class Yogi(Optimizer):
    """(Same Yogi implementation as before)"""
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
        # SWITCH: Choose between Yogi and AdamW based on config
        if self.config.optimizer_name == "yogi":
            print(">>> Initializing Custom YOGI Optimizer")
            opt = Yogi(
                model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        else:
            print(">>> Initializing Standard ADAMW Optimizer")
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )

        # NeMo Compatibility Hook
        opt.mcore_optimizer = opt
        return [opt]

# --- 2. MAIN TRAINING LOOP ---

def main():
    parser = argparse.ArgumentParser(description="NeMo GPT Training with Real Data")
    parser.add_argument("--optimizer", type=str, default="yogi", choices=["yogi", "adamw"], help="Choose optimizer")
    parser.add_argument("--data_prefix", type=str, default="my_real_data", help="Prefix of the .bin/.idx files")
    parser.add_argument("--max_steps", type=int, default=100, help="Training steps")
    args = parser.parse_args()

    # 1. Tokenizer (Must match what you used in preprocessing step)
    # If you used 'gpt2' in preprocessing, load 'gpt2' here.
    tokenizer = AutoTokenizer(pretrained_model_name="gpt2")

    # 2. Model Configuration (Small GPT for testing)
    model_config = llm.GPTConfig(
        num_layers=4,          # Reduced for speed
        hidden_size=512,       # Reduced for speed
        ffn_hidden_size=2048,
        num_attention_heads=8,
        seq_length=1024,
        init_method_std=0.02,
        make_vocab_size_divisible_by=128,
    )
    model = llm.GPTModel(config=model_config, tokenizer=tokenizer)

    # 3. Real Data Module
    # We look for files named {args.data_prefix}_text_document.bin / .idx
    data_path = f"{args.data_prefix}_text_document"
    
    data = PreTrainingDataModule(
        paths=[data_path],       # Path to binary data
        seq_length=1024,         # Must match model seq_length
        global_batch_size=8,     # Adjust based on GPU VRAM
        micro_batch_size=2,
        tokenizer=tokenizer,     # Pass the tokenizer object
        num_workers=0            # 0 for safe debugging, increase for speed
    )

    # 4. Strategy (Single GPU for testing)
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ddp="megatron",
    )

    # 5. Optimizer Configuration
    opt_config = OptimizerConfig(
        lr=6e-4,                 # Slightly lower LR for real data
        betas=(0.9, 0.95),       # Standard LLM betas
        eps=1e-8,
        weight_decay=0.1,
        optimizer_name=args.optimizer
    )
    optimizer = CustomOptimizerModule(config=opt_config)

    # 6. Trainer
    trainer = nl.Trainer(
        devices=1,
        accelerator="gpu",
        strategy=strategy,
        max_steps=args.max_steps,
        precision="bf16-mixed",
        log_every_n_steps=5,
        enable_checkpointing=False, # Disable to save disk space for tests
    )

    print(f"\n{'='*40}\nStarting Training with {args.optimizer.upper()}\n{'='*40}\n")
    
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        optim=optimizer,
        log=nl.NeMoLogger(name=f"gpt_{args.optimizer}_test", log_dir="results")
    )

if __name__ == "__main__":
    main()