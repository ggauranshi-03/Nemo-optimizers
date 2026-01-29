import math
import argparse
import os
import torch
from torch.optim.optimizer import Optimizer
import torch.optim as optim

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

        # Print detailed report every few steps
        # We access the first param state to get the step count
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
#                      Muon Optimizer Diagnostic Callback                      #
# ============================================================================ #
class MuonOptimizerDiagnosticCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            for i, opt in enumerate(trainer.optimizers):
                if hasattr(opt, 'state'):
                    muon_count = 0
                    adamw_count = 0
                    
                    if len(opt.state) == 0:
                        print(f"\n[DIAGNOSTIC CRITICAL] Optimizer {i} has EMPTY state after step 0!")
                        print("This means p.grad (and p.main_grad) was None for all parameters.")
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

# ============================================================================ #
#                      Adam Optimizer Diagnostic Callback                      #
# ============================================================================ #
class AdamOptimizerDiagnosticCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        for i, opt in enumerate(trainer.optimizers):
            if hasattr(opt, 'param_groups'):
                opt_name = opt.__class__.__name__
                lr = opt.param_groups[0]['lr']
                eps = opt.param_groups[0].get('eps', 'N/A')
                print(f"\n{'='*70}")
                print(f"[DIAGNOSTIC] Optimizer {i}:")
                print(f"  Class: {opt_name}  <-- SHOULD BE 'AdamW'")
                print(f"  LR: {lr}")
                print(f"  Eps: {eps}")
                print(f"{'='*70}\n")

# ============================================================================ #
#                       Muon Layer-Wise Diagnostic Callback                    #
# ============================================================================ #
class LayerWiseDiagnosticCallback(Callback):
    """
    Prints exactly which optimizer (Muon or AdamW) is assigned to each named parameter.
    """
    def on_train_start(self, trainer, pl_module):
        print(f"\n{'='*100}")
        print(f"{'[LAYER-WISE OPTIMIZER ASSIGNMENT]':^100}")
        print(f"{'='*100}")
        print(f"{'PARAMETER NAME':<60} | {'SHAPE':<15} | {'ASSIGNED OPTIMIZER'}")
        print("-" * 100)

        muon_count = 0
        adam_count = 0

        # Iterate over all named parameters in the model
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue

            # --- REPLICATING MUON LOGIC ---
            # 1. Embeddings (Large vocab size > 10,000 rows)
            is_embedding = (param.ndim == 2 and param.size(0) > 10000)
            
            # 2. Linear Weights (2D matrices that are NOT embeddings) -> Muon
            is_linear_weight = (param.ndim == 2 and not is_embedding)

            if is_linear_weight:
                optim_type = "MUON"
                muon_count += 1
            else:
                optim_type = "ADAMW (Aux)"
                adam_count += 1

            print(f"{name:<60} | {str(list(param.shape)):<15} | {optim_type}")

        print("-" * 100)
        print(f"SUMMARY: Muon Layers: {muon_count} | AdamW Layers: {adam_count}")
        print(f"{'='*100}\n")

# ============================================================================ #
#                             Training Function                                #
# ============================================================================ #
def run_training(optim_type: str, args):
    exp_name = f"{args.name}_{optim_type}"
    exp_base_dir = os.path.join(args.exp_dir, exp_name)
    checkpoint_dir = os.path.join(exp_base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

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

    # 1. Initialize the optimizer module ONLY for muon
    optim_module = None
    if optim_type == "muon":
        optim_module = MuonOptimizerModule(
            lr=args.muon_lr,
            adam_w_lr=args.adamw_lr,
            weight_decay=args.weight_decay
        )
        callbacks = [
            PerplexityCallback(), 
            MuonOptimizerDiagnosticCallback(),
            LayerWiseDiagnosticCallback()
        ]
        print(f"\n{'='*70}")
        print(f"[START] Muon Training (Custom Optimizer)")
        print(f"{'='*70}\n")
    else:  # adam
        # No optim_module assigned here so NeMo uses its default
        callbacks = [
            PerplexityCallback(),
            AdamOptimizerDiagnosticCallback()
        ]
        print(f"\n{'='*70}")
        print(f"[START] AdamW Training (NeMo Default Optimizer)")
        print(f"{'='*70}\n")

    # 2. Pass the optim_module to the model (it will be None for Adam)
    model = llm.GPTModel(
        config=model_config, 
        tokenizer=tokenizer,
        optim=optim_module  # When None, NeMo defaults to its internal AdamW
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
    )

    loggers = []
    if args.enable_wandb:
        wandb_logger = WandbLogger(
            name=exp_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
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
        save_top_k=1,
        every_n_train_steps=100,
        save_weights_only=True,
    )

    callbacks.append(checkpoint_callback)

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
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=None,
        optim=None,
        resume=nl.AutoResume(resume_if_exists=False),
    )

    # Post-training: Access saved checkpoints
    print(f"\n{'='*70}")
    print(f"[TRAINING COMPLETE: {optim_type.upper()}]")
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

# ============================================================================ #
#                                Main Function                                 #
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(description="NeMo GPT Pretraining with Muon and AdamW")
    parser.add_argument("--name", type=str, default="gpt_optim_comparison", help="Experiment name")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="Experiments directory")
    parser.add_argument("--wandb_project", type=str, default="nemo-gpt-optim-comparison", help="WandB Project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB Entity (User/Team)")
    parser.add_argument("--wandb_offline", action="store_true", help="Run WandB offline")
    parser.add_argument("--enable_wandb", action="store_true", default=True)
    
    # Training Config
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus_per_node", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--global_batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    
    # Optimizer Config
    parser.add_argument("--optimizer", type=str, default="both", choices=["muon", "adam", "both"], help="Optimizer to use: muon (hybrid), adam (pure AdamW), or both")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="Muon LR for hybrid")
    parser.add_argument("--adamw_lr", type=float, default=0.003, help="AdamW LR for hybrid aux params")
    parser.add_argument("--pure_adam_lr", type=float, default=0.001, help="LR for pure AdamW")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    
    args = parser.parse_args()

    if args.optimizer in ["both", "muon"]:
        run_training("muon", args)
    
    if args.optimizer in ["both", "adam"]:
        run_training("adam", args)

if __name__ == "__main__":
    main()
