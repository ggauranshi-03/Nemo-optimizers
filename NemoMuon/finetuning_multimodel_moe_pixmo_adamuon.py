# ── stdlib / third-party ─────────────────────────────────────────────────────
import math
import os
import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import requests
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# ── NeMo 2.0 / Lightning ─────────────────────────────────────────────────────
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from lightning.pytorch.strategies import DDPStrategy
from nemo import lightning as nl
from nemo.lightning.pytorch.optim import OptimizerModule

from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================
#  CONFIG DATACLASSES
# ============================================================
@dataclass
class ModelConfig:
    num_layers:           int   = 4
    hidden_size:          int   = 512  # Matches CLIP-ViT-B/32
    num_attention_heads:  int   = 8
    ffn_hidden_size:      int   = 2048
    num_moe_experts:      int   = 8
    top_k:                int   = 2
    img_size:             int   = 224  # Standard CLIP size
    text_seq_len:         int   = 77   # CLIP max token length
    aux_loss_weight:      float = 0.01

@dataclass
class DataConfig:
    pixmo_split:          str   = "train"
    pixmo_max_samples:    int   = 1000
    text_seq_len:         int   = 256
    min_freq:             int   = 5
    cache_dir:            str   = "./pixmo_cache"
    num_download_workers: int   = 16
    batch_size:           int   = 16
    num_workers:          int   = 4

@dataclass
class OptimConfig:
    lr:            float = 0.001
    adam_lr:       float = 0.003
    weight_decay:  float = 0.0
    betas:         Any   = field(default_factory=lambda: (0.9, 0.95))
    ns_steps:      int   = 5

@dataclass
class TrainerConfig:
    max_steps:     int   = 100
    log_every:     int   = 10
    use_amp:       bool  = True
    exp_dir:       str   = "experiments"
    wandb_project: str   = "nemo-multimodal-moe-muon"
    name:          str   = "multimodal_moe_nemo"
    enable_wandb:  bool  = True


# ============================================================
#  DATASET & LIGHTNING DATAMODULE: handles downloading, vocabulary building, tokenization, image transformation, and formatting the data
# ============================================================
class PixmoCapDataset(Dataset):
    def __init__(self, split="train", max_samples=1000, cache_dir="./pixmo_cache"):
        super().__init__()
        self.ds = load_dataset("allenai/pixmo-cap", split=split)
        if max_samples > 0:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
        ])

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        caption = self.ds[idx]["caption"]
        # Placeholder for image loading logic from your original cache
        img = Image.new("RGB", (224, 224), (128, 128, 128)) 
        img_tensor = self.image_transform(img)

        # Use CLIP Tokenizer
        inputs = self.processor(text=[caption], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        
        return {
            "image": img_tensor,
            "input_ids": inputs.input_ids.squeeze(0),
        }

# handles data loading on GPUs, settings for training
class PixmoDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.dataset = PixmoCapDataset(max_samples=self.cfg.data.pixmo_max_samples)
        self.vocab_size = 49408 # CLIP official vocab size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.data.batch_size, shuffle=True, num_workers=4)


# ============================================================
#  MODEL ARCHITECTURE (PyTorch)
# ============================================================
class PatchEmbed(nn.Module): # convert the 2D image into a 1D sequence
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class SparseMoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        logits = self.router(x_flat)
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        gates = F.softmax(top_k_logits, dim=-1)
        
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx, gate = indices[:, k], gates[:, k]
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    out[mask] += gate[mask].unsqueeze(-1) * self.experts[e](x_flat[mask])
        
        # Simple Load Balance Loss
        probs = F.softmax(logits, dim=-1)
        aux_loss = self.num_experts * (probs.mean(0) * probs.mean(0)).sum()
        return out.reshape(B, S, D), aux_loss

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, experts=8, top_k=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = SparseMoE(dim, mlp_dim, experts, top_k)

    def forward(self, x):
        res, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + res
        res, aux = self.moe(self.norm2(x))
        return x + res, aux

# class MultimodalMoEBackbone(nn.Module):
#     def __init__(self, cfg, vocab_size: int):
#         super().__init__()
#         D = cfg.hidden_size
#         self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, 3, D)
#         self.text_embed = nn.Embedding(vocab_size, D)
#         self.cls_token  = nn.Parameter(torch.zeros(1, 1, D))
#         self.pos_embed  = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches + cfg.text_seq_len, D))

#         self.blocks = nn.ModuleList([
#             TransformerBlock(D, cfg.num_attention_heads, cfg.ffn_hidden_size, cfg.num_moe_experts, cfg.top_k, cfg.dropout)
#             for _ in range(cfg.num_layers)
#         ])
#         self.norm = nn.LayerNorm(D)
#         self.lm_head = nn.Linear(D, vocab_size)

#     def forward(self, images, input_ids):
#         B = images.size(0)
#         img_tokens, txt_tokens = self.patch_embed(images), self.text_embed(input_ids)
#         x = torch.cat([self.cls_token.expand(B, -1, -1), img_tokens, txt_tokens], dim=1)
#         x = x + self.pos_embed[:, :x.size(1), :]

#         total_aux_loss = x.new_zeros(1).squeeze()
#         for blk in self.blocks:
#             x, aux_loss = blk(x)
#             total_aux_loss += aux_loss

#         text_feats = self.norm(x)[:, 1 + self.patch_embed.num_patches:, :]
#         return self.lm_head(text_feats), total_aux_loss

# ============================================================
#  FOUNDATION BACKBONE
# ============================================================
class CLIPMoEBackbone(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze CLIP to focus training on the MoE adapter
        for p in self.clip.parameters(): p.requires_grad = False

        D = cfg.hidden_size
        self.blocks = nn.ModuleList([
            TransformerBlock(D, cfg.num_attention_heads, cfg.ffn_hidden_size, cfg.num_moe_experts, cfg.top_k)
            for _ in range(cfg.num_layers)
        ])
        self.lm_head = nn.Linear(D, vocab_size)

    def forward(self, images, input_ids):
        # Extract features from pre-trained encoders
        img_feats = self.clip.get_image_features(pixel_values=images) # [B, 512]
        txt_feats = self.clip.get_text_features(input_ids=input_ids)   # [B, 512]
        
        # Combine into a sequence for the MoE blocks
        x = torch.stack([img_feats, txt_feats], dim=1) 
        
        total_aux = x.new_zeros(1).squeeze()
        for blk in self.blocks:
            x, aux = blk(x)
            total_aux += aux
            
        return self.lm_head(x[:, 1, :]), total_aux
# ============================================================
#  ADAMUON OPTIMIZER MODULE
# ============================================================
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    transposed = G.size(0) > G.size(1)
    if transposed: G = G.t()
    X = (G / (G.norm() + eps)).bfloat16()
    for _ in range(steps):
        A = X.t() @ X
        X = X @ (a * torch.eye(X.size(1), device=X.device, dtype=X.dtype) + b * A + c * A @ A)
    return (X.t() if transposed else X).float()

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

class AdaMuonOptimizerModule(OptimizerModule):
    def __init__(self, lr: float, adam_w_lr: float, weight_decay: float, lr_scheduler=None):
        super().__init__(lr_scheduler=lr_scheduler)
        self.lr = lr
        self.adam_w_lr = adam_w_lr
        self.weight_decay = weight_decay

    def optimizers(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        return [AdaMuon(params, lr=self.lr, adam_w_lr=self.adam_w_lr, weight_decay=self.weight_decay)]


# ============================================================
#  MODEL WRAPPER : complete, end-to-end Multimodal
# ============================================================
class MultimodalMoEModel(pl.LightningModule):
    def __init__(self, cfg, vocab_size, optim_module):
        super().__init__()
        self.cfg = cfg
        self.backbone = CLIPMoEBackbone(cfg.model, vocab_size)
        self.optim_module = optim_module
        self.criterion = nn.CrossEntropyLoss()

    # def training_step(self, batch, idx):
    #     logits, aux = self.backbone(batch["image"], batch["input_ids"])
    #     loss = self.criterion(logits, batch["input_ids"]) + self.cfg.model.aux_loss_weight * aux
    #     self.log("train_loss", loss, prog_bar=True)
    #     return loss
    def training_step(self, batch, idx):
        # logits shape: [Batch, VocabSize]
        # targets shape: [Batch, 77]
        logits, aux = self.backbone(batch["image"], batch["input_ids"])
        
        # FIX: Compare the prediction to the first token (index 0) of the sequence
        # or use a specific label if this is classification.
        targets = batch["input_ids"][:, 0] # Takes the first token from each sequence
        
        loss = self.criterion(logits, targets) + self.cfg.model.aux_loss_weight * aux
        
        # Add accuracy logging for better monitoring
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self): return self.optim_module.optimizers(self)


# ============================================================
#  CALLBACKS
# ============================================================
class PerplexityCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None:
            pl_module.log("train_perplexity", torch.exp(loss.detach()), prog_bar=True)

class OptimizerDiagnosticCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            for opt in trainer.optimizers:
                muon_count = sum(1 for s in opt.state.values() if s.get('use_muon', False))
                adam_count = len(opt.state) - muon_count
                print(f"\n{'='*70}\n[PARAMETER CLASSIFICATION]\n  AdaMuon layers: {muon_count}\n  AdamW layers: {adam_count}\n{'='*70}\n")

class LayerWiseDiagnosticCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print(f"\n{'='*100}\n{'[LAYER-WISE OPTIMIZER ASSIGNMENT]':^100}\n{'='*100}")
        print(f"{'PARAMETER NAME':<60} | {'SHAPE':<15} | {'ASSIGNED OPTIMIZER'}\n" + "-"*100)
        for name, p in pl_module.named_parameters():
            if not p.requires_grad: continue
            optim_type = "ADAMUON" if (p.ndim == 2 and p.size(0) <= 10000) else "ADAMW (Aux)"
            print(f"{name:<60} | {str(list(p.shape)):<15} | {optim_type}")
        print("=" * 100 + "\n")


# ============================================================
#  MAIN LOOP (NeMo 2.0 API style)
# ============================================================
def build_config(args):
    return OmegaConf.create({
        "model": {
            "num_layers": args.num_layers, "hidden_size": args.hidden_size,
            "num_attention_heads": args.num_attention_heads, "ffn_hidden_size": args.ffn_hidden_size,
            "num_moe_experts": args.num_moe_experts, "top_k": args.top_k,
            "img_size": 32, "patch_size": 4, "text_seq_len": args.text_seq_len,
            "dropout": 0.0, "aux_loss_weight": 0.01,
        },
        "data": {
            "pixmo_split": "train", "pixmo_max_samples": args.pixmo_max_samples,
            "text_seq_len": args.text_seq_len, "min_freq": 5, "cache_dir": args.cache_dir,
            "num_download_workers": 16, "batch_size": args.global_batch_size, "num_workers": 4,
        },
        "optim": { "lr": args.lr, "adam_lr": args.adam_lr, "weight_decay": args.weight_decay }
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="finetune_multimodal_moe_adamuon")
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("--wandb_project", type=str, default="nemo-multimodal")
    parser.add_argument("--enable_wandb", action="store_true", default=True)
    parser.add_argument("--cache_dir", type=str, default="./pixmo_cache")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--ffn_hidden_size", type=int, default=3072)
    parser.add_argument("--num_moe_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--text_seq_len", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--global_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--adam_lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pixmo_max_samples", type=int, default=1000)
    args = parser.parse_args()

    # cfg = build_config(args)
    cfg = OmegaConf.create({
        "model": ModelConfig().__dict__,
        "data": {"pixmo_max_samples": 1000, "batch_size": 8},
        "optim": { "lr": args.lr, "adam_lr": args.adam_lr, "weight_decay": args.weight_decay }
    })
    os.makedirs(args.exp_dir, exist_ok=True)

    # 1. Init Data
    data = PixmoDataModule(cfg)
    data.setup()

    # 2. Init Optimizer Module
    # optim = AdaMuonOptimizerModule(lr=cfg.optim.lr, adam_w_lr=cfg.optim.adam_lr, weight_decay=cfg.optim.weight_decay)
    optim = AdaMuonOptimizerModule(lr=0.0001, adam_w_lr=0.00005,weight_decay=cfg.optim.weight_decay)


    # 3. Init Model Wrapper
    model = MultimodalMoEModel(cfg, vocab_size=data.vocab_size, optim_module=optim)

    loggers = [WandbLogger(project=args.wandb_project, name=args.name, save_dir=args.exp_dir)] if args.enable_wandb and WANDB_AVAILABLE else None

    # 4. NeMo Lightning Trainer (Using 'auto' strategy to handle pure PyTorch models safely)
    trainer = nl.Trainer(
        devices="auto",
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed",
        log_every_n_steps=1,
        logger=loggers,
        callbacks=[
            ModelCheckpoint(dirpath=os.path.join(args.exp_dir, "checkpoints"), save_last=True),
            PerplexityCallback(),
            OptimizerDiagnosticCallback(),
            LayerWiseDiagnosticCallback()
        ],
        gradient_clip_val=1.0,
    )

    print(f"\n{'='*70}\n[START] Multimodal MoE Training (NeMo 2.0 Native)\n{'='*70}\n")
    
    trainer.fit(model=model, datamodule=data)

if __name__ == "__main__":
    main()
