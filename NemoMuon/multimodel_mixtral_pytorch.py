#!/usr/bin/env python
import math
import argparse
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ================================================================
#              CIFAR-10 MULTIMODAL DATASET (IMAGE + TEXT)
# ================================================================

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class ImageCaptionDataset(Dataset):
    """
    CIFAR-10 images + simple text caption: "a photo of <class_name>".
    Text is tokenized with a tiny custom vocab, no external tokenizer.
    """

    def __init__(self, root, train, seq_length=1024, text_seq_len=6):
        self.cifar = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]),
        )
        self.seq_length = seq_length
        self.text_seq_len = text_seq_len

        # Build tiny vocab from all possible captions
        sentences = ["a photo of an object" for _ in CIFAR10_CLASSES]
        vocab = {"<pad>": 0}
        for sent in sentences:
            for w in sent.strip().split():
                if w not in vocab:
                    vocab[w] = len(vocab)
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]

        # Precompute captions per class
        self.class_to_tokens = {}
        for idx, name in enumerate(CIFAR10_CLASSES):
            sent = f"a photo of {name}"
            toks = self.text_to_ids(sent)
            self.class_to_tokens[idx] = toks

    def text_to_ids(self, text):
        words = text.strip().split()
        ids = []
        for w in words:
            ids.append(self.vocab.get(w, self.pad_id))
        if len(ids) < self.text_seq_len:
            ids = ids + [self.pad_id] * (self.text_seq_len - len(ids))
        else:
            ids = ids[:self.text_seq_len]
        return ids

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, label = self.cifar[idx]  # img: (3,32,32)
        text_ids = self.class_to_tokens[label]

        text_ids = torch.tensor(text_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return {
            "image": img,          # (3,32,32)
            "text_ids": text_ids,  # (T,)
            "label": label,        # scalar
        }


# ================================================================
#                      VISION PATCH EMBEDDING
# ================================================================

class PatchEmbed(nn.Module):
    """2D image to patch embedding, ViT-style."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)              # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


# ================================================================
#                      DENSE MoE LAYER (NO TOP-K)
# ================================================================

class DenseMoE(nn.Module):
    """
    Simple dense MoE:
      y = sum_e softmax(router(x))[e] * expert_e(x)
    Experts are independent MLPs; router is a linear layer.
    """

    def __init__(self, dim, hidden_dim, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: (B, S, D)
        return: (B, S, D)
        """
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)

        logits = self.router(x_flat)           # (N, E)
        gates = F.softmax(logits, dim=-1)      # (N, E)

        out = 0.0
        for e, expert in enumerate(self.experts):
            y_e = expert(x_flat)               # (N, D)
            gate_e = gates[:, e:e+1]           # (N, 1)
            out = out + gate_e * y_e

        out = out.reshape(B, S, D)
        return out


# ================================================================
#                      TRANSFORMER BLOCK with MoE
# ================================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, num_experts=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.moe = DenseMoE(dim, mlp_dim, num_experts=num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        x_res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_res + self.dropout(attn_out)

        # MoE MLP
        y_res = x
        y_norm = self.norm2(x)
        moe_out = self.moe(y_norm)
        y = y_res + self.dropout(moe_out)

        return y


# ================================================================
#                 MULTIMODAL MoE MODEL (IMAGE + TEXT)
# ================================================================

class MultimodalMoEModel(nn.Module):
    """
    - Vision: CIFAR-10 image -> patch embeddings
    - Text: small vocab embedding of "a photo of <class>"
    - Sequence: [CLS] + image_patches + text_tokens
    - Backbone: Transformer with MoE FFN
    - Head: classification from CLS token
    """

    def __init__(
        self,
        num_layers=12,
        hidden_size=768,
        ffn_hidden_size=3072,
        num_attention_heads=12,
        num_experts=8,
        num_classes=10,
        img_size=32,
        patch_size=4,
        vocab_size=32,
        text_seq_len=6,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.text_seq_len = text_seq_len

        # Vision patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=hidden_size,
        )
        num_patches = self.patch_embed.num_patches

        # Text embedding
        self.text_embed = nn.Embedding(vocab_size, hidden_size)

        # CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.max_seq_len = 1 + num_patches + text_seq_len
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_seq_len, hidden_size)
        )

        # Transformer blocks with MoE
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_size,
                num_heads=num_attention_heads,
                mlp_dim=ffn_hidden_size,
                num_experts=num_experts,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images, text_ids):
        """
        images: (B,3,32,32)
        text_ids: (B,T)
        """
        B = images.size(0)

        img_tokens = self.patch_embed(images)         # (B, Np, D)
        txt_tokens = self.text_embed(text_ids)        # (B, T, D)
        

        cls_token = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls_token, img_tokens, txt_tokens], dim=1)  # (B,S,D)

        S = x.size(1)
        x = x + self.pos_embed[:, :S, :]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls = x[:, 0]                  # (B,D)
        logits = self.head(cls)        # (B,num_classes)
        return logits


# ================================================================
#                     Muon Math Helper Functions
# ================================================================

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
        X = X @ (
            a * torch.eye(X.size(1), device=X.device, dtype=X.dtype) + B
        )

    if transposed:
        X = X.t()

    return X.float()


# ================================================================
#                         AdaMuon Optimizer
# ================================================================

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
            eps=eps,
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

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            muon_beta1, muon_beta2 = group["betas"]
            ns_steps = group["ns_steps"]
            adam_lr = group["adam_w_lr"]
            adam_beta1, adam_beta2 = group["adam_w_betas"]
            eps = group["eps"]

            for p in group["params"]:
                grad = p.grad
                if grad is None and hasattr(p, "main_grad"):
                    grad = p.main_grad
                if grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["use_muon"] = self._classify_param(p)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                use_muon = state["use_muon"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step_t = state["step"]

                if use_muon:
                    muon_updates += 1
                    exp_avg.mul_(muon_beta1).add_(grad, alpha=1 - muon_beta1)
                    M_t = exp_avg
                    O_t = zeropower_via_newtonschulz5(M_t, steps=ns_steps)
                    exp_avg_sq.mul_(muon_beta2).addcmul_(
                        O_t, O_t, value=1 - muon_beta2
                    )
                    bias_correction2 = 1 - muon_beta2**step_t
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
                    exp_avg_sq.mul_(adam_beta2).addcmul_(
                        grad, grad, value=1 - adam_beta2
                    )
                    bias_correction1 = 1 - adam_beta1**step_t
                    bias_correction2 = 1 - adam_beta2**step_t
                    step_size = adam_lr / bias_correction1
                    bias_correction2_sqrt = math.sqrt(bias_correction2)
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        step_count = 0
        if (
            len(self.param_groups) > 0
            and len(self.param_groups[0]["params"]) > 0
        ):
            p0 = self.param_groups[0]["params"][0]
            if p0 in self.state:
                step_count = self.state[p0]["step"]

        if step_count % self.log_interval == 0 or step_count == 1:
            print(f"\n[OPTIMIZER CHECK step {step_count}]")
            print(f"  > AdaMuon Updates: {muon_updates}")
            print(f"  > AdamW Updates: {adam_updates}")

        return loss


# ================================================================
#                         TRAINING LOOP
# ================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Dataset + Loader
    train_dataset = ImageCaptionDataset(
        root=args.data_dir,
        train=True,
        seq_length=args.seq_length,
        text_seq_len=args.text_seq_len,
    )

    vocab_size = len(train_dataset.vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = MultimodalMoEModel(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_experts=args.num_moe_experts,
        num_classes=10,
        img_size=32,
        patch_size=4,
        vocab_size=vocab_size,
        text_seq_len=args.text_seq_len,
        dropout=0.0,
    )

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Optimizer: AdaMuon
    optimizer = AdaMuon(
        model.parameters(),
        lr=args.lr,
        adam_w_lr=args.adam_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        ns_steps=5,
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # WandB
    if args.enable_wandb and WANDB_AVAILABLE:
        os.makedirs(args.exp_dir, exist_ok=True)
        wandb.init(
            project=args.wandb_project,
            name=args.name,
            dir=args.exp_dir,
            config=vars(args),
        )
    elif args.enable_wandb and not WANDB_AVAILABLE:
        print("[WARN] wandb not installed, disabling wandb logging.")
        args.enable_wandb = False

    model.train()
    step = 0
    epoch = 0

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.max_steps:
                break

            images = batch["image"].to(device, non_blocking=True)
            text_ids = batch["text_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=torch.bfloat16):
                logits = model(images, text_ids)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = logits.max(dim=1)
            acc = (preds == labels).float().mean().item()

            if step % args.log_every == 0:
                print(
                    f"[STEP {step:05d}] "
                    f"loss={loss.item():.4f} "
                    f"acc={acc*100:.2f}%"
                )
                if args.enable_wandb:
                    wandb.log({"train/loss": loss.item(), "train/acc": acc}, step=step)

            step += 1

    if args.enable_wandb:
        wandb.finish()


# ================================================================
#                            MAIN
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure PyTorch Multimodal MoE on CIFAR-10 with AdaMuon"
    )
    parser.add_argument("--name", type=str, default="multimodal_moe_adamuon_pytorch")
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("--wandb_project", type=str, default="nemo-multimodal-moe-muon")
    parser.add_argument("--enable_wandb", action="store_true", default=False)

    parser.add_argument("--data_dir", type=str, default="./data")

    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--ffn_hidden_size", type=int, default=3072)
    parser.add_argument("--num_moe_experts", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--text_seq_len", type=int, default=6)

    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--global_batch_size", type=int, default=16)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--adam_lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--log_every", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)
    train(args)
