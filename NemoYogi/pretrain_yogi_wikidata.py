import math
import argparse
from dataclasses import dataclass
from typing import Optional
import os

import torch
import torch.nn.functional as F

# --- Standard NeMo Imports ---
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.gpt.data import PreTrainingDataModule

# --- REAL IMPORTS from PyTorch Lightning ---
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from torch.optim.optimizer import Optimizer


# ============================================================================ #
#                           Perplexity Callback                               #
# ============================================================================ #
class PerplexityCallback(Callback):
    """Computes and logs perplexity (exp(loss))"""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = None
        if isinstance(outputs, dict):
            loss = outputs.get("loss") or outputs.get("reduced_train_loss")
        elif torch.is_tensor(outputs):
            loss = outputs
        
        if loss is not None:
            ppl = torch.exp(loss.detach())
            pl_module.log("train_perplexity", ppl, prog_bar=True, on_step=True, on_epoch=False)


# ============================================================================ #
#                        Yogi Optimizer Implementation                        #
# ============================================================================ #
class Yogi(Optimizer):
    """Yogi optimizer"""
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-3,
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

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach().float()
                if grad.is_sparse:
                    raise RuntimeError("Yogi does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.full_like(p, fill_value=1e-6, dtype=torch.float32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                p_data_fp32 = p.data.float()
                if group["weight_decay"] != 0:
                    grad = grad.add(p_data_fp32, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_sq = grad.mul(grad)
                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_sq),
                    grad_sq,
                    value=-(1 - beta2),
                )

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = group["lr"] / bias_correction1

                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(p_data_fp32.to(p.data.dtype))

        return loss


# ============================================================================ #
#              DIAGNOSTIC CALLBACK - Verify Optimizer is Yogi                #
# ============================================================================ #
class OptimizerDiagnosticCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        for i, opt in enumerate(trainer.optimizers):
            if hasattr(opt, 'param_groups'):
                opt_name = opt.__class__.__name__
                lr = opt.param_groups[0]['lr']
                eps = opt.param_groups[0].get('eps', 'N/A')
                print(f"\n{'='*70}")
                print(f"[DIAGNOSTIC] Optimizer {i}:")
                print(f"  Class: {opt_name}  <-- SHOULD BE 'Yogi'")
                print(f"  LR: {lr}")
                print(f"  Eps: {eps}")
                print(f"{'='*70}\n")


# ============================================================================ #
#                               Main Function                                 #
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(
        description="NeMo GPT Pretraining with Yogi Optimizer"
    )
    # Experiment Config
    parser.add_argument("--name", type=str, default="gpt_yogi_pretrain", help="Experiment name")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="Experiments directory")
    
    # WandB Config
    parser.add_argument("--wandb_project", type=str, default="nemo-gpt-yogi", help="WandB Project Name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB Entity (User/Team)")
    parser.add_argument("--wandb_offline", action="store_true", help="Run WandB in offline mode")
    parser.add_argument("--enable_wandb", action="store_true", default=True, help="Enable WandB logging")

    # Training Config
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--num_gpus_per_node", type=int, default=2, help="GPUs per node")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Attention heads")
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--max_steps", type=int, default=150, help="Max training steps")
    parser.add_argument("--global_batch_size", type=int, default=16, help="Global batch size")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size")
    
    # Optimizer Config
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Yogi")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Yogi")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_distributed_optimizer", action="store_true", default=False)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("[CONFIG] Training Parameters:")
    print(f"  Experiment: {args.name}")
    print(f"  Experiment Dir: {args.exp_dir}")
    print(f"  LR: {args.lr}")
    print(f"  Beta1: {args.beta1}, Beta2: {args.beta2}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  WandB Enabled: {args.enable_wandb}")
    print(f"{'='*70}\n")

    # Create experiment directory structure
    exp_base_dir = os.path.join(args.exp_dir, args.name)
    checkpoint_dir = os.path.join(exp_base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"[INFO] Checkpoints will be saved to: {os.path.abspath(checkpoint_dir)}\n")

    # 1. Model Configuration
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
    model = llm.GPTModel(config=model_config, tokenizer=tokenizer)

    # 2. Data Module
    data = PreTrainingDataModule(
        paths={
            "train": [
                0.75, "data/wikitext103/my_gpt_data_text_document_text_document",
                0.25, "data/wikitext103/my_gpt_data_text_document_text_document",
            ],
            "validation": ["data/wikitext103/my_gpt_data_text_document_text_document"],
            "test": ["data/wikitext103/my_gpt_data_text_document_text_document"],
        },
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        num_workers=8,
        pin_memory=True,
    )

    # 3. Strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ddp="megatron",
        find_unused_parameters=False,
        use_distributed_optimizer=False,
        save_sharded_state_dict=False,  # ✅ CRITICAL: Custom optimizer fix
    )

    # 4. Setup Loggers
    loggers = []
    
    if args.enable_wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            offline=args.wandb_offline,
            log_model="all",  # Log all checkpoints to WandB
            save_dir=exp_base_dir,
        )
        loggers.append(wandb_logger)
        print(f"[INFO] WandB Logger initialized for project: {args.wandb_project}\n")

    # 5. Initialize Model Checkpoint Callback
    # ✅ FIX: Monitor correct metric and use absolute path
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-{step}-{reduced_train_loss:.4f}",
        monitor="reduced_train_loss",  # ✅ FIXED: Use correct metric
        mode="min",
        save_last=True,
        save_top_k=3,
        every_n_train_steps=50,
        save_weights_only=True,
    )

    print(f"[INFO] ModelCheckpoint callback configured:\n")
    print(f"  Save Directory: {checkpoint_dir}")
    print(f"  Monitor Metric: reduced_train_loss")
    print(f"  Save Top K: 3")
    print(f"  Every N Train Steps: 50")
    print(f"  Save Last: True\n")

    # 6. Trainer
    trainer = nl.Trainer(
        devices=args.num_gpus_per_node,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        precision="bf16-mixed",
        log_every_n_steps=10,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=loggers if loggers else None,  # ✅ FIXED: Pass loggers
        callbacks=[
            checkpoint_callback,
            PerplexityCallback(),
            OptimizerDiagnosticCallback(),
        ],
    )

    # 7. Resume configuration
    resume = nl.AutoResume(resume_if_exists=False)

    # ====================================================================== #
    # CRITICAL FIX: Override trainer's optimizer_class BEFORE training       #
    # ====================================================================== #
    def create_yogi_optimizer(param_groups):
        """Factory function NeMo will call to create the optimizer"""
        return Yogi(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=1e-3,
            weight_decay=args.weight_decay,
        )
    
    trainer._optimizer_class = Yogi
    print("[TRAINER] Set trainer._optimizer_class = Yogi\n")

    # 8. Train
    print(f"\n{'='*70}")
    print("[START] Beginning training...")
    print(f"{'='*70}\n")
    
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=None,
        optim=None,
        resume=resume,
    )

    # ====================================================================== #
    # Post-training: Access saved checkpoints
    # ====================================================================== #
    print(f"\n{'='*70}")
    print("[TRAINING COMPLETE]")
    print(f"{'='*70}")
    print(f"\n[CHECKPOINTS SAVED TO]: {os.path.abspath(checkpoint_dir)}\n")
    
    if hasattr(checkpoint_callback, 'best_model_path'):
        print(f"[BEST MODEL]: {checkpoint_callback.best_model_path}\n")
    
    # List all checkpoints
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        print(f"[CHECKPOINTS FOUND]: {len(checkpoints)}")
        for ckpt in sorted(checkpoints):
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            print(f"  - {ckpt} ({os.path.getsize(ckpt_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
