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

# ── NeMo 2.0 / Lightning ─────────────────────────────────────────────────────
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from nemo import lightning as nl
from nemo.lightning.pytorch.optim import OptimizerModule

from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset

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
    hidden_size:          int   = 192
    num_attention_heads:  int   = 8
    ffn_hidden_size:      int   = 3072
    num_moe_experts:      int   = 8
    top_k:                int   = 2
    num_classes:          int   = 10
    img_size:             int   = 32
    patch_size:           int   = 4
    vocab_size:           int   = 1000 # Filled at runtime
    text_seq_len:         int   = 256
    dropout:              float = 0.0
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
    def __init__(self, split="train", max_samples=1000, text_seq_len=256, min_freq=5, cache_dir="./pixmo_cache", num_download_workers=16):
        super().__init__()
        self.text_seq_len = text_seq_len
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        ds = load_dataset("allenai/pixmo-cap", split=split)
        if max_samples and max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.ds = ds

        from collections import Counter
        counter = Counter()
        for i in range(len(self.ds)):
            for w in self.ds[i]["caption"].strip().split():
                counter[w] += 1

        self.vocab = {"<pad>": 0, "<unk>": 1}
        for w, c in counter.items():
            if c >= min_freq and w not in self.vocab:
                self.vocab[w] = len(self.vocab)

        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]

        self.image_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        caption = self.ds[idx]["caption"]
        path = os.path.join(self.cache_dir, f"{idx:07d}.jpg")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (32, 32), (128, 128, 128))
            
        img_tensor = self.image_transform(img)

        words = caption.strip().split()
        full_ids = [self.vocab.get(w, self.unk_id) for w in words]
        if len(full_ids) < self.text_seq_len:
            full_ids += [self.pad_id] * (self.text_seq_len - len(full_ids))
        else:
            full_ids = full_ids[:self.text_seq_len]

        input_ids  = (full_ids[:-1] + [self.pad_id])[: self.text_seq_len]
        target_ids = (full_ids[1:]  + [self.pad_id])[: self.text_seq_len]

        return {
            "image":      img_tensor,
            "input_ids":  torch.tensor(input_ids,  dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

# handles data loading on GPUs, settings for training
class PixmoDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.dataset = PixmoCapDataset(
            split=self.cfg.data.pixmo_split,
            max_samples=self.cfg.data.pixmo_max_samples,
            text_seq_len=self.cfg.data.text_seq_len,
            min_freq=self.cfg.data.min_freq,
            cache_dir=self.cfg.data.cache_dir,
            num_download_workers=self.cfg.data.num_download_workers,
        )
        self.vocab_size = len(self.dataset.vocab)
        self.pad_id = self.dataset.pad_id

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.cfg.data.batch_size, shuffle=True,
            num_workers=self.cfg.data.num_workers, pin_memory=True, drop_last=True,
        )


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

class SparseMoE(nn.Module): # Mixture of Experts
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_noise = 0.01

        self.router  = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)

        router_logits = self.router(x_flat)
        if self.training: router_logits += torch.randn_like(router_logits) * self.router_noise

        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx, gate_k = top_k_indices[:, k], top_k_gates[:, k]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if not mask.any(): continue
                output[mask] += gate_k[mask].unsqueeze(-1) * self.experts[e](x_flat[mask])

        # Load-balance auxiliary loss
        router_probs = F.softmax(router_logits, dim=-1)
        one_hot_topk = torch.zeros_like(router_probs).scatter_(1, top_k_indices, 1.0)
        aux_loss = self.num_experts * (one_hot_topk.mean(dim=0) * router_probs.mean(dim=0)).sum()

        return output.reshape(B, S, D), aux_loss

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, num_experts=8, top_k=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.moe   = SparseMoE(dim, mlp_dim, num_experts=num_experts, top_k=top_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)
        return x, aux_loss

class MultimodalMoEBackbone(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        D = cfg.hidden_size
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, 3, D)
        self.text_embed = nn.Embedding(vocab_size, D)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed  = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches + cfg.text_seq_len, D))

        self.blocks = nn.ModuleList([
            TransformerBlock(D, cfg.num_attention_heads, cfg.ffn_hidden_size, cfg.num_moe_experts, cfg.top_k, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.norm = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, vocab_size)

    def forward(self, images, input_ids):
        B = images.size(0)
        img_tokens, txt_tokens = self.patch_embed(images), self.text_embed(input_ids)
        x = torch.cat([self.cls_token.expand(B, -1, -1), img_tokens, txt_tokens], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]

        total_aux_loss = x.new_zeros(1).squeeze()
        for blk in self.blocks:
            x, aux_loss = blk(x)
            total_aux_loss += aux_loss

        text_feats = self.norm(x)[:, 1 + self.patch_embed.num_patches:, :]
        return self.lm_head(text_feats), total_aux_loss


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

class MuonOptimizerModule(OptimizerModule):
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
    def __init__(self, cfg, vocab_size, pad_id, optim_module: OptimizerModule):
        super().__init__()
        self.cfg = cfg
        self._pad_id = pad_id
        self.optim_module = optim_module
        
        self.backbone = MultimodalMoEBackbone(cfg.model, vocab_size)
        self._lm_criterion = nn.CrossEntropyLoss(ignore_index=self._pad_id)

    def forward(self, images, input_ids):
        return self.backbone(images, input_ids)

    def training_step(self, batch, batch_idx):
        images, input_ids, target_ids = batch["image"], batch["input_ids"], batch["target_ids"]
        lm_logits, aux_loss = self(images, input_ids)
        
        lm_loss = self._lm_criterion(lm_logits.reshape(-1, lm_logits.size(-1)), target_ids.reshape(-1))
        loss = lm_loss + self.cfg.model.aux_loss_weight * aux_loss
        
        preds, mask = lm_logits.argmax(dim=-1), target_ids != self._pad_id
        acc = ((preds == target_ids) & mask).sum().float() / mask.sum().clamp(min=1)
            
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        return self.optim_module.optimizers(self)


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
                print(f"\n{'='*70}\n[PARAMETER CLASSIFICATION]\n  Muon layers: {muon_count}\n  AdamW layers: {adam_count}\n{'='*70}\n")

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
    parser.add_argument("--name", type=str, default="multimodal_moe_nemo")
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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--adam_lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pixmo_max_samples", type=int, default=1000)
    args = parser.parse_args()

    cfg = build_config(args)
    os.makedirs(args.exp_dir, exist_ok=True)

    # 1. Init Data
    data = PixmoDataModule(cfg)
    data.setup()

    # 2. Init Optimizer Module
    optim = MuonOptimizerModule(lr=cfg.optim.lr, adam_w_lr=cfg.optim.adam_lr, weight_decay=cfg.optim.weight_decay)

    # 3. Init Model Wrapper
    model = MultimodalMoEModel(cfg, vocab_size=data.vocab_size, pad_id=data.pad_id, optim_module=optim)

    loggers = [WandbLogger(project=args.wandb_project, name=args.name, save_dir=args.exp_dir)] if args.enable_wandb and WANDB_AVAILABLE else None

    # 4. NeMo Lightning Trainer (Using 'auto' strategy to handle pure PyTorch models safely)
    trainer = nl.Trainer(
        devices="auto",
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
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
    
    # Run the trainer fit (Note: llm.train is strictly for the llm.GPTModel collections)
    trainer.fit(model=model, datamodule=data)

if __name__ == "__main__":
    main()
