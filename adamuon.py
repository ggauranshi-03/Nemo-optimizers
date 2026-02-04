import math
import argparse
import os
import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer

# --- Standard NeMo Imports ---
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.lightning.pytorch.optim import OptimizerModule

# --- REAL IMPORTS from PyTorch Lightning ---
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

# ============================================================================ #
#                           Fixed AdaMuon Math Helper                          #
# ============================================================================ #
def orthogonalize_via_newtonschulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration to compute orthogonalization of gradient matrix.
    This is the FIXED version that preserves gradient information.
    """
    assert G.ndim == 2, f"Expected 2D tensor, got {G.ndim}D"
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Save original shape for later
    original_rows, original_cols = G.shape
    
    # Transpose if needed for better conditioning
    if original_rows > original_cols:
        G = G.t()
        transposed = True
        rows, cols = G.shape
    else:
        transposed = False
        rows, cols = G.shape
    
    # Initialize with normalized gradient (preserves gradient direction)
    norm = G.norm() + eps
    X = G / norm
    X = X.bfloat16()
    
    # Newton-Schulz iteration for orthogonalization
    for _ in range(steps):
        A = X.t() @ X
        B = b * A + c * A @ A
        X = X @ (a * torch.eye(cols, device=X.device, dtype=X.dtype) + B)
    
    # Scale by original gradient norm to preserve magnitude information
    X = X.float() * norm
    
    if transposed:
        X = X.t()
    
    # Restore to original dimensions if needed
    if X.shape != (original_rows, original_cols):
        X = X.view(original_rows, original_cols)
    
    return X

# ============================================================================ #
#                           Fixed AdaMuon Class                                #
# ============================================================================ #
class AdaMuon(Optimizer):
    """
    FIXED AdaMuon: Adaptive Muon Optimizer
    Fixed the critical issue: Preserves gradient information instead of using sign(g)
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, 
                 ns_steps=5, eps=1e-8, rank=None, world_size=None):
        
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required. For single GPU pass rank=0 and world_size=1.")
        
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, 
                       nesterov=nesterov, ns_steps=ns_steps, eps=eps)
        
        # Group parameters by size for efficient distributed updates
        params = list(params)
        param_groups = []
        
        for size in {p.numel() for p in params}:
            buf = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=buf, 
                update_buffer_views=[buf[i] for i in range(world_size)])
            param_groups.append(group)
        
        super().__init__(param_groups, defaults)
        
        self.log_interval = 10
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            eps = group["eps"]
            handle = None
            params_world = None

            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world), alpha=-group["lr"])

            # AdaMuon Distributed Sharding Logic
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]

                    # Handle gradient from Megatron
                    g = p.grad
                    if g is None and hasattr(p, 'main_grad'):
                        g = p.main_grad
                    if g is None:
                        g = torch.zeros_like(p)

                    state = self.state[p]

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    if "step" not in state:
                        state["step"] = 0
                    state["step"] += 1

                    buf: Tensor = state["momentum_buffer"]

                    # Momentum update
                    buf.mul_(group["momentum"]).add_(g, alpha=1 - group["momentum"])

                    # Nesterov momentum if enabled
                    if group['nesterov']:
                        g = g.add(buf, alpha=group["momentum"])
                    else:
                        g = buf

                    # FIXED: Apply Newton-Schulz orthogonalization to the GRADIENT
                    # Reshape for matrix operations if needed
                    original_shape = g.shape
                    if g.ndim > 2:
                        g_reshaped = g.view(g.size(0), -1)
                    else:
                        g_reshaped = g

                    # Apply orthogonalization to gradient
                    g_ortho = orthogonalize_via_newtonschulz(g_reshaped, steps=group["ns_steps"])

                    # Reshape back to original shape
                    if g.ndim > 2:
                        g_ortho = g_ortho.view(original_shape)

                    # Adaptive scaling
                    rows, cols = (p.shape[0], p.shape[1]) if p.ndim >= 2 else (1, p.numel())
                    scale = max(1, rows / cols) ** 0.5
                    g_ortho.mul_(scale)

                    # Convert to buffer dtype
                    g = g_ortho.to(update_buffer.dtype)

                    # CRITICAL FIX: Ensure tensor is contiguous before all_gather
                    g = g.contiguous()

                else:
                    g = update_buffer_views[self.rank]

                if base_i > 0:
                    update_prev()

                # CRITICAL FIX: Flatten g before all_gather and ensure it's contiguous
                g_flat = g.flatten().contiguous()

                handle = dist.all_gather_into_tensor(update_buffer, g_flat, async_op=True)
                params_world = params[base_i : base_i + self.world_size]

            if params: 
                update_prev()

        return loss

# ============================================================================ #
#                           Hybrid Optimizer Wrapper                           #
# ============================================================================ #
class HybridOptimizer(Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)
        defaults = optimizers[0].defaults if optimizers else {}
        super().__init__(self.param_groups, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'hybrid_optimizers': [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        if 'hybrid_optimizers' in state_dict:
            opt_states = state_dict['hybrid_optimizers']
            for opt, s_dict in zip(self.optimizers, opt_states):
                opt.load_state_dict(s_dict)
        else:
            super().load_state_dict(state_dict)

# ============================================================================ #
#                           Optimizer Module Wrapper                           #
# ============================================================================ #
class AdaMuonOptimizerModule(OptimizerModule):
    def __init__(self, lr: float, adam_w_lr: float, weight_decay: float, lr_scheduler=None):
        super().__init__(lr_scheduler=lr_scheduler)
        # Fix for NeMo Megatron Strategy check
        self.config = None 
        
        self.lr = lr
        self.adam_w_lr = adam_w_lr
        self.weight_decay = weight_decay

    def optimizers(self, model):
        params_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
        
        muon_params = []
        decay_params = []
        nodecay_params = []

        for name, p in params_dict.items():
            is_embedding = (p.ndim == 2 and p.size(0) > 10000)
            is_linear_weight = (p.ndim >= 2 and not is_embedding)

            if is_linear_weight:
                muon_params.append(p)
            elif p.ndim < 2:
                nodecay_params.append(p)
            else:
                decay_params.append(p)

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        optim_groups_adam = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        opt_adam = torch.optim.AdamW(optim_groups_adam, lr=self.adam_w_lr, betas=(0.9, 0.95))

        opt_adamuon = AdaMuon(
            muon_params, 
            lr=self.lr, 
            weight_decay=self.weight_decay,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            rank=rank, 
            world_size=world_size
        )

        print(f"\n{'='*70}")
        print(f"Initialized FIXED Hybrid Optimizer:")
        print(f"  > AdaMuon params: {len(muon_params)} tensors")
        print(f"  > AdamW (Decay) params: {len(decay_params)} tensors")
        print(f"  > AdamW (No Decay) params: {len(nodecay_params)} tensors")
        print(f"{'='*70}")

        return [HybridOptimizer([opt_adam, opt_adamuon])]

# ============================================================================ #
#                           Callbacks                                          #
# ============================================================================ #
class PerplexityCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = None
        if isinstance(outputs, dict):
            loss = outputs.get("loss") or outputs.get("reduced_train_loss")
        elif torch.is_tensor(outputs):
            loss = outputs

        if loss is not None:
            ppl = torch.exp(loss.detach())
            pl_module.log("train_perplexity", ppl, prog_bar=True, on_step=True, on_epoch=False)

class OptimizerDiagnosticCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            print(f"\n{'='*70}")
            print(f"[OPTIMIZER DIAGNOSTIC]")
            hybrid_opt = trainer.optimizers[0]
            if isinstance(hybrid_opt, HybridOptimizer):
                for i, opt in enumerate(hybrid_opt.optimizers):
                    name = opt.__class__.__name__
                    total_params = sum([len(g['params']) for g in opt.param_groups])
                    print(f"  Internal Optimizer {i}: {name} (Tensors: {total_params})")
            else:
                print("  [WARNING] Optimizer is not HybridOptimizer instance.")
            print(f"{'='*70}\n")

class AdaMuonDebugCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Debug callback to monitor AdaMuon updates"""
        if batch_idx % 10 == 0:
            hybrid_opt = trainer.optimizers[0]
            if isinstance(hybrid_opt, HybridOptimizer):
                for i, opt in enumerate(hybrid_opt.optimizers):
                    if isinstance(opt, AdaMuon):
                        ada_muon_updates = 0
                        ada_muon_norm = 0.0
                        for group in opt.param_groups:
                            for p in group['params']:
                                state = opt.state[p]
                                if 'step' in state and state['step'] > 0:
                                    ada_muon_updates += 1
                        print(f"[AdaMuon Debug] Steps: {batch_idx}, Updated tensors: {ada_muon_updates}")

class LayerWiseDiagnosticCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print(f"\n{'='*100}")
        print(f"{'[LAYER-WISE OPTIMIZER ASSIGNMENT]':^100}")
        print(f"{'='*100}")
        print(f"{'PARAMETER NAME':<60} | {'SHAPE':<15} | {'ASSIGNED OPTIMIZER'}")
        print("-" * 100)
        muon_count = 0
        adam_count = 0
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            is_embedding = (param.ndim == 2 and param.size(0) > 10000)
            is_linear_weight = (param.ndim >= 2 and not is_embedding)
            if is_linear_weight:
                optim_type = "AdaMuon (FIXED)"
                muon_count += 1
            else:
                optim_type = "AdamW"
                adam_count += 1
            print(f"{name:<60} | {str(list(param.shape)):<15} | {optim_type}")
        print("-" * 100)
        print(f"SUMMARY: AdaMuon Layers: {muon_count} | AdamW Layers: {adam_count}")
        print(f"{'='*100}\n")

# ============================================================================ #
#                                 Main Function                                #
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(description="NeMo GPT Pretraining with FIXED AdaMuon")
    parser.add_argument("--name", type=str, default="gpt_adamuon", help="Experiment name")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="Experiments directory")
    parser.add_argument("--wandb_project", type=str, default="nemo-gpt-muon", help="WandB Project")
    parser.add_argument("--wandb_offline", action="store_true", help="Run WandB offline")
    parser.add_argument("--enable_wandb", action="store_true", default=True)

    # Training Config
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus_per_node", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=100) 
    
    # Batch Size - divisible by GPU count
    parser.add_argument("--global_batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=1)

    # Optimizer Config - Adjusted learning rates
    parser.add_argument("--lr", type=float, default=0.0003, help="AdaMuon LR")  # Reduced from 0.0006
    parser.add_argument("--adam_lr", type=float, default=0.003, help="AdamW LR")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    args = parser.parse_args()

    exp_base_dir = os.path.join(args.exp_dir, args.name)
    checkpoint_dir = os.path.join(exp_base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_config = llm.GPTConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.hidden_size * 4,
        num_attention_heads=args.num_attention_heads,
        seq_length=args.seq_length,
    )

    optimizer_arg = AdaMuonOptimizerModule(
        lr=args.lr,
        adam_w_lr=args.adam_lr,
        weight_decay=args.weight_decay
    )

    tokenizer = AutoTokenizer(pretrained_model_name="gpt2")

    model = llm.GPTModel(
        config=model_config,
        tokenizer=tokenizer,
        optim=optimizer_arg
    )

    data = PreTrainingDataModule(
        paths={
            "train": [1.0, "data/wikitext103/my_gpt_data_text_document_text_document"],
            "validation": ["data/wikitext103/my_gpt_data_text_document_text_document"],
            "test": ["data/wikitext103/my_gpt_data_text_document_text_document"],
        },
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        num_workers=2,
        pin_memory=False,
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ddp="megatron",  
        find_unused_parameters=False,
        use_distributed_optimizer=False, 
        gradient_as_bucket_view=True,
    )

    loggers = []
    if args.enable_wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.wandb_project,
            offline=args.wandb_offline,
            save_dir=exp_base_dir,
        )
        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-{step}-{reduced_train_loss:.4f}",
        monitor="reduced_train_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
        every_n_train_steps=50,
        save_weights_only=True,
    )

    trainer = nl.Trainer(
        devices=args.num_gpus_per_node,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        precision="bf16-mixed",
        log_every_n_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=loggers if loggers else None,
        callbacks=[
            checkpoint_callback,
            PerplexityCallback(),
            OptimizerDiagnosticCallback(),
            AdaMuonDebugCallback(),
            LayerWiseDiagnosticCallback()
        ],
        gradient_clip_val=1.0,
    )

    print(f"\n{'='*70}")
    print(f"[START] FIXED AdaMuon Training (Hybrid: Fixed AdaMuon + AdamW)")
    print(f"Changes made:")
    print(f"1. Removed torch.sign() - now using actual gradients")
    print(f"2. Fixed Newton-Schulz to preserve gradient magnitude")
    print(f"3. Added proper scaling")
    print(f"4. Increased max_steps to {args.max_steps}")
    print(f"5. Reduced AdaMuon LR to {args.lr}")
    print(f"{'='*70}\n")

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=None,
        optim=None,
        resume=nl.AutoResume(resume_if_exists=False),
    )

if __name__ == "__main__":
    main()
