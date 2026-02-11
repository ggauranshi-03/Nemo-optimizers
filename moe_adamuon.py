import math
import argparse
import os
import torch
from torch.optim.optimizer import Optimizer
# This must happen BEFORE importing NeMo/Megatron
import sys

# Monkey patch the problematic module
def patch_megatron_moe():
    """Patch Megatron-Core to handle missing Transformer Engine"""
    try:
        from megatron.core.transformer.moe import moe_utils
        
        # Store the original function
        original_router_gating_linear = moe_utils.router_gating_linear
        
        def patched_router_gating_linear(inp, weight, bias=None, router_dtype=None):
            """Patched version that doesn't use Transformer Engine"""
            # Just use standard PyTorch linear operation
            if router_dtype is not None and router_dtype != inp.dtype:
                inp = inp.to(router_dtype)
                weight = weight.to(router_dtype)
                if bias is not None:
                    bias = bias.to(router_dtype)
            
            output = torch.nn.functional.linear(inp, weight, bias)
            return output
        
        # Replace the function
        moe_utils.router_gating_linear = patched_router_gating_linear
        print("[PATCH] Successfully patched Megatron-Core MoE router to bypass Transformer Engine")
        
    except Exception as e:
        print(f"[PATCH] Warning: Could not patch Megatron-Core: {e}")

# Apply patch after megatron is imported but before it's used
def delayed_patch():
    """Apply patch after imports"""
    patch_megatron_moe()

# --- Standard NeMo Imports ---
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.lightning.pytorch.optim import OptimizerModule

# Apply the patch now that megatron is loaded
delayed_patch()

# --- REAL IMPORTS from PyTorch Lightning ---
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

# ============================================================================ #
#                           Muon Math Helper Functions                         #
# ============================================================================ #
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
            lr = group['lr']
            weight_decay = group['weight_decay']
            muon_beta1, muon_beta2 = group['betas']
            ns_steps = group['ns_steps']
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
                    exp_avg.mul_(muon_beta1).add_(grad, alpha=1 - muon_beta1)
                    M_t = exp_avg
                    O_t = zeropower_via_newtonschulz5(M_t, steps=ns_steps)
                    exp_avg_sq.mul_(muon_beta2).addcmul_(O_t, O_t, value=1 - muon_beta2)
                    bias_correction2 = 1 - muon_beta2 ** step_t
                    v_hat = exp_avg_sq / bias_correction2
                    denom = v_hat.sqrt().add_(eps)
                    O_hat = O_t / denom
                    rms = O_hat.pow(2).mean().sqrt()
                    scaling_factor = 0.2 / (rms + eps)
                    update_term = O_hat.mul_(scaling_factor)
                    if weight_decay != 0:
                        update_term.add_(p, alpha=weight_decay)
                    p.add_(update_term, alpha=-lr)
                else:
                    adam_updates += 1
                    if weight_decay != 0:
                        p.mul_(1 - adam_lr * weight_decay)
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
            print(f"  > AdaMuon Updates: {muon_updates}")
            print(f"  > AdamW Updates: {adam_updates}")

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
        params = [p for p in model.parameters() if p.requires_grad]
        return [AdaMuon(         
            params,
            lr=self.lr,
            adam_w_lr=self.adam_w_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),   
            ns_steps=5
        )]

# ============================================================================ #
#                            Perplexity Callback                               #
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

# ============================================================================ #
#                           Diagnostic Callback                                #
# ============================================================================ #
class OptimizerDiagnosticCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            for i, opt in enumerate(trainer.optimizers):
                if hasattr(opt, 'state'):
                    muon_count = 0
                    adamw_count = 0
                    
                    if len(opt.state) == 0:
                        print(f"\n[DIAGNOSTIC] Optimizer {i} has EMPTY state!")
                        continue

                    for param, s in opt.state.items():
                        if s.get('use_muon', False):
                            muon_count += 1
                        else:
                            adamw_count += 1
                    
                    print(f"\n{'='*70}")
                    print(f"[PARAMETER CLASSIFICATION]")
                    print(f"  Muon layers: {muon_count}")
                    print(f"  AdamW layers: {adamw_count}")
                    print(f"{'='*70}\n")

class LayerWiseDiagnosticCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        
        print(f"\n{'='*100}")
        print(f"[MODEL PARAMETER COUNT]")
        print(f"  Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
        print(f"  Trainable Parameters: {trainable_params:,} (~{trainable_params/1e6:.1f}M)")
        print(f"{'='*100}\n")

# ============================================================================ #
#                                Main Function                                 #
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(description="NeMo MoE Pretraining with AdaMuon")
    parser.add_argument("--name", type=str, default="moe_adamuon", help="Experiment name")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="Experiments directory")
    parser.add_argument("--wandb_project", type=str, default="nemo-moe-muon", help="WandB Project")
    parser.add_argument("--wandb_offline", action="store_true", help="Run WandB offline")
    parser.add_argument("--enable_wandb", action="store_true", default=True)
    
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus_per_node", type=int, default=4)
    
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--ffn_hidden_size", type=int, default=3072)
    parser.add_argument("--num_moe_experts", type=int, default=8)
    parser.add_argument("--moe_router_topk", type=int, default=2)
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=0.01)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--global_batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--adam_lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    args = parser.parse_args()

    exp_base_dir = os.path.join(args.exp_dir, args.name)
    checkpoint_dir = os.path.join(exp_base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_config = llm.MixtralConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_attention_heads,
        seq_length=args.seq_length,
        num_moe_experts=args.num_moe_experts,
        moe_router_topk=args.moe_router_topk,
        moe_aux_loss_coeff=args.moe_aux_loss_coeff,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        apply_rope_fusion=False,
        gradient_accumulation_fusion=False,
        bias_activation_fusion=False,
        masked_softmax_fusion=False,
        persist_layer_norm=False,
        recompute_method='block',
        recompute_num_layers=1,
    )

    optimizer_arg = MuonOptimizerModule(
        lr=args.lr,
        adam_w_lr=args.adam_lr,
        weight_decay=args.weight_decay
    )

    tokenizer = AutoTokenizer(pretrained_model_name="gpt2")
    
    model = llm.MixtralModel(
        config=model_config, 
        tokenizer=tokenizer,
        optim=optimizer_arg
    )

    print(f"\n{'='*70}")
    print(f"[MODEL CONFIG]")
    print(f"  Layers: {args.num_layers}, Hidden: {args.hidden_size}")
    print(f"  Experts: {args.num_moe_experts}, Top-K: {args.moe_router_topk}")
    print(f"  Seq Length: {args.seq_length}")
    print(f"{'='*70}\n")

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
        num_workers=8,
        pin_memory=True,
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ddp="megatron",
        find_unused_parameters=False,
        use_distributed_optimizer=False,
        expert_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
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
            LayerWiseDiagnosticCallback()
        ],
        gradient_clip_val=1.0,
    )

    print(f"\n{'='*70}")
    print(f"[START] MoE Training with AdaMuon")
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
