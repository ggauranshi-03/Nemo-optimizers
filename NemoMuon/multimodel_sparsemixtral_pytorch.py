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
from datasets import load_dataset
import requests
from io import BytesIO
from PIL import Image

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


# class ImageCaptionDataset(Dataset):
#     """
#     CIFAR-10 images + simple text caption: "a photo of <class_name>".
#     Text is tokenized with a tiny custom vocab, no external tokenizer.
#     """

#     def __init__(self, root, train, seq_length=1024, text_seq_len=6):
#         self.cifar = datasets.CIFAR10(
#             root=root,
#             train=train,
#             download=True,
#             transform=transforms.Compose([
#                 transforms.Resize((32, 32)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.4914, 0.4822, 0.4465],
#                     std=[0.2023, 0.1994, 0.2010],
#                 ),
#             ]),
#         )
#         self.seq_length = seq_length
#         self.text_seq_len = text_seq_len

#         # Build tiny vocab from all possible captions
#         sentences = ["a photo of an object" for _ in CIFAR10_CLASSES]
#         vocab = {"<pad>": 0}
#         for sent in sentences:
#             for w in sent.strip().split():
#                 if w not in vocab:
#                     vocab[w] = len(vocab)
#         self.vocab = vocab
#         self.pad_id = vocab["<pad>"]

#         # Precompute captions per class
#         self.class_to_tokens = {}
#         for idx, name in enumerate(CIFAR10_CLASSES):
#             sent = f"a photo of {name}"
#             toks = self.text_to_ids(sent)
#             self.class_to_tokens[idx] = toks

#     def text_to_ids(self, text):
#         words = text.strip().split()
#         ids = []
#         for w in words:
#             ids.append(self.vocab.get(w, self.pad_id))
#         if len(ids) < self.text_seq_len:
#             ids = ids + [self.pad_id] * (self.text_seq_len - len(ids))
#         else:
#             ids = ids[:self.text_seq_len]
#         return ids

#     def __len__(self):
#         return len(self.cifar)

#     def __getitem__(self, idx):
#         img, label = self.cifar[idx]  # img: (3,32,32)
#         text_ids = self.class_to_tokens[label]

#         text_ids = torch.tensor(text_ids, dtype=torch.long)
#         label = torch.tensor(label, dtype=torch.long)

#         return {
#             "image": img,          # (3,32,32)
#             "text_ids": text_ids,  # (T,)
#             "label": label,        # scalar
#         }


# class PixmoCapDataset(Dataset):
#     """
#     PixMo-Cap dense captioning dataset:
#       - uses allenai/pixmo-cap
#       - each example has an image_url and a very long 'caption'
#       - we build a simple word-level vocab over captions
#       - returns (image, input_ids, target_ids) for LM loss
#     """

#     def __init__(
#         self,
#         split: str = "train",
#         max_samples: int = 100_000,
#         text_seq_len: int = 256,
#         min_freq: int = 5,
#     ):
#         super().__init__()
#         self.text_seq_len = text_seq_len
#         self._fail_count = 0
#         self._total_count = 0

#         print(f"[PIXMO-CAP] Loading allenai/pixmo-cap ({split})...")
#         ds = load_dataset("allenai/pixmo-cap", split=split)
#         if max_samples is not None and max_samples > 0:
#             ds = ds.select(range(min(max_samples, len(ds))))
#         self.ds = ds

#         # Build vocab from caption text on a subset
#         print("[PIXMO-CAP] Building vocab from captions...")
#         from collections import Counter
#         counter = Counter()

#         sample_for_vocab = min(100_000, len(self.ds))
#         for i in range(sample_for_vocab):
#             cap = self.ds[i]["caption"]
#             for w in cap.strip().split():
#                 counter[w] += 1

#         self.vocab = {"<pad>": 0, "<unk>": 1}
#         for w, c in counter.items():
#             if c >= min_freq and w not in self.vocab:
#                 self.vocab[w] = len(self.vocab)

#         self.pad_id = self.vocab["<pad>"]
#         self.unk_id = self.vocab["<unk>"]

#         # Image preprocessing (resize to your ViT input size)
#         self.image_transform = transforms.Compose([
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.5, 0.5, 0.5],
#                 std=[0.5, 0.5, 0.5],
#             ),
#         ])

#         print(f"[PIXMO-CAP] Vocab size: {len(self.vocab)}")
#         print(f"[PIXMO-CAP] Dataset size: {len(self.ds)}")

#     def __len__(self):
#         return len(self.ds)

#     def _load_image_from_url(self, url: str):
#         headers = {
#             "User-Agent": (
#                 "Mozilla/5.0 (X11; Linux x86_64) "
#                 "AppleWebKit/537.36 (KHTML, like Gecko) "
#                 "Chrome/120.0.0.0 Safari/537.36"
#             )
#         }
#         for attempt in range(3):
#             try:
#                 resp = requests.get(url, timeout=8, headers=headers)
#                 resp.raise_for_status()
#                 img = Image.open(BytesIO(resp.content))
#                 # Fix PIL palette+transparency warning
#                 if img.mode in ("P", "PA"):
#                     img = img.convert("RGBA")
#                 img = img.convert("RGB")
#                 return img, False  # (image, failed=False)
#             except Exception as e:
#                 if attempt == 2:  # last attempt
#                     print(
#                         f"[PIXMO-CAP] ⚠ Failed after 3 attempts | "
#                         f"URL: {url[:80]} | Error: {e}"
#                     )
#                 else:
#                     time.sleep(0.5 * (attempt + 1))  # 0.5s, 1.0s backoff

#         return Image.new("RGB", (32, 32), color="gray"), True  # failed=True


#     def get_image_fail_rate(self):
#         if self._total_count == 0:
#             return 0.0
#         return self._fail_count / self._total_count * 100


#     def text_to_ids(self, text: str):
#         words = text.strip().split()
#         ids = []
#         for w in words:
#             ids.append(self.vocab.get(w, self.unk_id))
#         if len(ids) < self.text_seq_len:
#             ids = ids + [self.pad_id] * (self.text_seq_len - len(ids))
#         else:
#             ids = ids[: self.text_seq_len]
#         return ids

#     def __getitem__(self, idx):
#         row = self.ds[idx]
#         url = row["image_url"]
#         caption = row["caption"]

#         # Try loading image, mark as failed if fallback used
#         pil_img, img_failed = self._load_image_from_url(url)
#         img_tensor = self.image_transform(pil_img)

#         full_ids = self.text_to_ids(caption)
#         input_ids = full_ids[:-1]
#         target_ids = full_ids[1:]
#         if len(input_ids) < self.text_seq_len:
#             input_ids  = input_ids  + [self.pad_id] * (self.text_seq_len - len(input_ids))
#             target_ids = target_ids + [self.pad_id] * (self.text_seq_len - len(target_ids))
#         else:
#             input_ids  = input_ids[:self.text_seq_len]
#             target_ids = target_ids[:self.text_seq_len]

#         return {
#             "image":      img_tensor,
#             "input_ids":  torch.tensor(input_ids,  dtype=torch.long),
#             "target_ids": torch.tensor(target_ids, dtype=torch.long),
#             "img_failed": torch.tensor(int(img_failed), dtype=torch.long),  # 0 or 1
#         }

class PixmoCapDataset(Dataset):
    """
    PixMo-Cap dense captioning dataset.
    Pre-downloads all images at init time using a thread pool.
    After init, __getitem__ never makes network requests.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int = 100_000,
        text_seq_len: int = 256,
        min_freq: int = 5,
        cache_dir: str = "./pixmo_cache",
        num_download_workers: int = 16,
    ):
        super().__init__()
        self.text_seq_len = text_seq_len
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        print(f"[PIXMO-CAP] Loading allenai/pixmo-cap ({split})...")
        ds = load_dataset("allenai/pixmo-cap", split=split)
        if max_samples is not None and max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.ds = ds

        # ── Build vocab ──────────────────────────────────────────────────
        print("[PIXMO-CAP] Building vocab from captions...")
        from collections import Counter
        counter = Counter()
        for i in range(len(self.ds)):
            cap = self.ds[i]["caption"]
            for w in cap.strip().split():
                counter[w] += 1

        self.vocab = {"<pad>": 0, "<unk>": 1}
        for w, c in counter.items():
            if c >= min_freq and w not in self.vocab:
                self.vocab[w] = len(self.vocab)

        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]

        print(f"[PIXMO-CAP] Vocab size: {len(self.vocab)}")

        # ── Image transform ──────────────────────────────────────────────
        self.image_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # ── Pre-download all images ──────────────────────────────────────
        self._pre_download_images(num_download_workers)

        print(f"[PIXMO-CAP] Dataset ready. Size: {len(self.ds)}")

    # ── Download helpers ─────────────────────────────────────────────────

    def _cache_path(self, idx: int) -> str:
        return os.path.join(self.cache_dir, f"{idx:07d}.jpg")

    def _download_one(self, idx: int) -> bool:
        """Download image for sample idx. Returns True on success."""
        path = self._cache_path(idx)
        if os.path.exists(path):
            return True  # already cached

        url = self.ds[idx]["image_url"]
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=10, headers=headers)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                if img.mode in ("P", "PA"):
                    img = img.convert("RGBA")
                img = img.convert("RGB")
                img.save(path, format="JPEG", quality=90)
                return True
            except Exception:
                if attempt < 2:
                    time.sleep(0.3 * (attempt + 1))
        # Save blank gray image as fallback so __getitem__ never fails
        Image.new("RGB", (32, 32), color=(128, 128, 128)).save(path, format="JPEG")
        return False

    def _pre_download_images(self, num_workers: int):
        """Download all images in parallel. Shows a progress bar."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        indices = list(range(len(self.ds)))
        # Skip already-cached
        to_download = [i for i in indices if not os.path.exists(self._cache_path(i))]

        if not to_download:
            print(f"[PIXMO-CAP] All {len(self.ds)} images already cached.")
            return

        print(
            f"[PIXMO-CAP] Pre-downloading {len(to_download)}/{len(self.ds)} images "
            f"using {num_workers} threads..."
        )

        fail_count = 0
        done_count = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._download_one, i): i for i in to_download}
            for future in as_completed(futures):
                success = future.result()
                done_count += 1
                if not success:
                    fail_count += 1
                if done_count % 500 == 0 or done_count == len(to_download):
                    print(
                        f"[PIXMO-CAP] Downloaded {done_count}/{len(to_download)} "
                        f"| Failures (blank fallback): {fail_count} "
                        f"({fail_count/len(to_download)*100:.1f}%)"
                    )

        print(
            f"[PIXMO-CAP] Pre-download complete. "
            f"Failures: {fail_count}/{len(to_download)} "
            f"({fail_count/len(to_download)*100:.1f}%) — saved as gray fallback."
        )

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self):
        return len(self.ds)

    def text_to_ids(self, text: str):
        words = text.strip().split()
        ids = [self.vocab.get(w, self.unk_id) for w in words]
        if len(ids) < self.text_seq_len:
            ids = ids + [self.pad_id] * (self.text_seq_len - len(ids))
        else:
            ids = ids[: self.text_seq_len]
        return ids

    def __getitem__(self, idx):
        row = self.ds[idx]
        caption = row["caption"]

        # Load from local cache — NEVER makes a network request
        path = self._cache_path(idx)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        img_tensor = self.image_transform(img)

        # LM next-token prediction
        full_ids = self.text_to_ids(caption)
        input_ids  = full_ids[:-1] + [self.pad_id]
        target_ids = full_ids[1:]  + [self.pad_id]
        input_ids  = input_ids[:self.text_seq_len]
        target_ids = target_ids[:self.text_seq_len]

        return {
            "image":      img_tensor,
            "input_ids":  torch.tensor(input_ids,  dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
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

# class DenseMoE(nn.Module):
#     """
#     Simple dense MoE:
#       y = sum_e softmax(router(x))[e] * expert_e(x)
#     Experts are independent MLPs; router is a linear layer.
#     """

#     def __init__(self, dim, hidden_dim, num_experts=8):
#         super().__init__()
#         self.num_experts = num_experts
#         self.router = nn.Linear(dim, num_experts)
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(dim, hidden_dim),
#                 nn.GELU(),
#                 nn.Linear(hidden_dim, dim),
#             )
#             for _ in range(num_experts)
#         ])

#     def forward(self, x):
#         """
#         x: (B, S, D)
#         return: (B, S, D)
#         """
#         B, S, D = x.shape
#         x_flat = x.reshape(B * S, D)

#         logits = self.router(x_flat)           # (N, E)
#         gates = F.softmax(logits, dim=-1)      # (N, E)

#         out = 0.0
#         for e, expert in enumerate(self.experts):
#             y_e = expert(x_flat)               # (N, D)
#             gate_e = gates[:, e:e+1]           # (N, 1)
#             out = out + gate_e * y_e

#         out = out.reshape(B, S, D)
#         return out


class SparseMoE(nn.Module):
    """
    Sparse Top-K MoE (as used in Mixtral, Switch Transformer, Molmo).
    
    For each token:
      1. Router produces logits over num_experts
      2. Top-K experts are selected
      3. Only those K experts run on that token
      4. Outputs are weighted sum of the K expert outputs
      5. Auxiliary load-balancing loss encourages uniform expert usage
    """

    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2, dropout=0.0):
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: linear projection to expert logits
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Each expert is an independent MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_experts)
        ])

        # Optional: small noise for router exploration during training
        self.router_noise = 0.01

    def forward(self, x):
        """
        x: (B, S, D)
        returns: output (B, S, D), aux_loss scalar
        """
        B, S, D = x.shape
        N = B * S
        x_flat = x.reshape(N, D)   # (N, D)

        # ── Router ────────────────────────────────────────────────────
        router_logits = self.router(x_flat)   # (N, E)

        # Add noise during training to prevent router collapse
        if self.training:
            noise = torch.randn_like(router_logits) * self.router_noise
            router_logits = router_logits + noise

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # top_k_logits: (N, K), top_k_indices: (N, K)

        # Softmax over selected K logits only (not all E)
        top_k_gates = F.softmax(top_k_logits, dim=-1)   # (N, K)

        # ── Dispatch tokens to experts ─────────────────────────────────
        # Build output by accumulating K expert outputs per token
        output = torch.zeros_like(x_flat)   # (N, D)

        # For each of the K slots
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]   # (N,) — which expert each token goes to
            gate_k = top_k_gates[:, k]         # (N,) — weight for this slot

            # Group tokens by expert for efficient batching
            for e in range(self.num_experts):
                token_mask = (expert_idx == e)   # (N,) bool
                if not token_mask.any():
                    continue
                tokens_for_e = x_flat[token_mask]             # (n_e, D)
                expert_out = self.experts[e](tokens_for_e)    # (n_e, D)
                gate_weight = gate_k[token_mask].unsqueeze(-1) # (n_e, 1)
                output[token_mask] += gate_weight * expert_out

        output = output.reshape(B, S, D)

        # ── Load-balancing auxiliary loss ──────────────────────────────
        # Encourages uniform distribution of tokens across experts.
        # loss_aux = num_experts * sum_e(f_e * P_e)
        # f_e = fraction of tokens routed to expert e
        # P_e = mean router probability for expert e
        router_probs = F.softmax(router_logits, dim=-1)    # (N, E) — no noise
        # f_e: fraction of tokens where expert e is in top-k
        one_hot_topk = torch.zeros_like(router_probs)
        one_hot_topk.scatter_(1, top_k_indices, 1.0)      # (N, E)
        f_e = one_hot_topk.mean(dim=0)                    # (E,)
        P_e = router_probs.mean(dim=0)                    # (E,)
        aux_loss = self.num_experts * (f_e * P_e).sum()   # scalar

        return output, aux_loss


# ================================================================
#                      TRANSFORMER BLOCK with MoE
# ================================================================

# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_dim, num_experts=8, dropout=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             batch_first=True,
#             dropout=dropout,
#         )
#         self.norm2 = nn.LayerNorm(dim)
#         self.moe = DenseMoE(dim, mlp_dim, num_experts=num_experts)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # Self-attention
#         x_res = x
#         x_norm = self.norm1(x)
#         attn_out, _ = self.attn(x_norm, x_norm, x_norm)
#         x = x_res + self.dropout(attn_out)

#         # MoE MLP
#         y_res = x
#         y_norm = self.norm2(x)
#         moe_out = self.moe(y_norm)
#         y = y_res + self.dropout(moe_out)

#         return y


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, num_experts=8, top_k=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.moe = SparseMoE(dim, mlp_dim, num_experts=num_experts, top_k=top_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        x_res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_res + self.dropout(attn_out)

        # Sparse MoE FFN
        y_res = x
        y_norm = self.norm2(x)
        moe_out, aux_loss = self.moe(y_norm)   # ← now returns aux_loss too
        y = y_res + self.dropout(moe_out)

        return y, aux_loss   # ← propagate aux_loss up


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
        top_k=2,  
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
        self.head = nn.Linear(hidden_size, num_classes)

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
        # self.blocks = nn.ModuleList([
        #     TransformerBlock(
        #         dim=hidden_size,
        #         num_heads=num_attention_heads,
        #         mlp_dim=ffn_hidden_size,
        #         num_experts=num_experts,
        #         dropout=dropout,
        #     )
        #     for _ in range(num_layers)
        # ])

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_size,
                num_heads=num_attention_heads,
                mlp_dim=ffn_hidden_size,
                num_experts=num_experts,
                top_k=top_k,           # ← NEW
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)          # classification (unused for PixMo)
        self.lm_head = nn.Linear(hidden_size, vocab_size)        # dense captioning LM head
        

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
    def _encode_sequence(self, x):
        """Shared backbone. Returns (encoded, total_aux_loss)."""
        S = x.size(1)
        x = x + self.pos_embed[:, :S, :]
        total_aux_loss = 0.0
        for blk in self.blocks:
            x, aux_loss = blk(x)
            total_aux_loss = total_aux_loss + aux_loss
        x = self.norm(x)
        return x, total_aux_loss

    def forward_caption(self, images, input_ids):
        B = images.size(0)
        img_tokens = self.patch_embed(images)
        txt_tokens = self.text_embed(input_ids)
        cls_token  = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, img_tokens, txt_tokens], dim=1)

        x, aux_loss = self._encode_sequence(x)   # ← unpack aux_loss

        text_start = 1 + self.patch_embed.num_patches
        text_feats = x[:, text_start:, :]
        logits = self.lm_head(text_feats)
        return logits, aux_loss   # ← return both


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

def get_grad_norm(model):
    """Compute total L2 gradient norm across all parameters."""
    total_norm = 0.0
    num_params_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params_with_grad += 1
    total_norm = total_norm ** 0.5
    return total_norm, num_params_with_grad

# ================================================================
#                         TRAINING LOOP
# ================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # PixMo-Cap dense captioning dataset
        # PixMo-Cap dense captioning dataset
    train_dataset = PixmoCapDataset(
        split="train",
        max_samples=args.pixmo_max_samples,
        text_seq_len=args.text_seq_len,
        cache_dir=args.cache_dir,
        num_download_workers=16,
    )

    vocab_size = len(train_dataset.vocab)
    pad_id = train_dataset.pad_id

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
        top_k=args.top_k,
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
    # criterion = nn.CrossEntropyLoss()
    lm_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)


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

    AUX_LOSS_WEIGHT = 0.01

    model.train()
    step = 0
    epoch = 0

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.max_steps:
                break

            images     = batch["image"].to(device, non_blocking=True)
            input_ids  = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ── Forward pass ──────────────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=torch.bfloat16):
                if isinstance(model, nn.DataParallel):
                    logits, aux_loss = model.module.forward_caption(images, input_ids)
                else:
                    logits, aux_loss = model.forward_caption(images, input_ids)

                lm_loss = lm_criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                )
                loss = lm_loss + AUX_LOSS_WEIGHT * aux_loss

            # ── Backward + optimizer step ─────────────────────────────────
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ── Logging (only every log_every steps) ─────────────────────
            if step % args.log_every == 0:

                # Token accuracy
                with torch.no_grad():
                    preds   = logits.argmax(dim=-1)           # (B, L)
                    mask    = target_ids != pad_id
                    correct = ((preds == target_ids) & mask).sum().item()
                    total   = mask.sum().item()
                    acc     = correct / total if total > 0 else 0.0

                # Gradient norm
                grad_norm, n_params_with_grad = get_grad_norm(model)

                # AMP scaler health
                scaler_scale = scaler.get_scale()

                print(
                    f"[STEP {step:05d}] "
                    f"loss={loss.item():.4f}  "
                    f"lm_loss={lm_loss.item():.4f}  "
                    f"aux_loss={aux_loss.item():.4f}  "
                    f"token_acc={acc*100:.2f}%  "
                    f"grad_norm={grad_norm:.4f}  "
                    f"params_w_grad={n_params_with_grad}  "
                    f"scaler_scale={scaler_scale}"
                )

                if args.enable_wandb:
                    wandb.log({
                        "train/loss":         loss.item(),
                        "train/lm_loss":      lm_loss.item(),
                        "train/aux_loss":     aux_loss.item(),
                        "train/token_acc":    acc,
                        "train/grad_norm":    grad_norm,
                        "train/scaler_scale": scaler_scale,
                    }, step=step)

            step += 1

    if args.enable_wandb:
        wandb.finish()

# ================================================================
#                            MAIN
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",             type=str,   default="multimodal_moe_adamuon_pytorch")
    parser.add_argument("--exp_dir",          type=str,   default="experiments")
    parser.add_argument("--wandb_project",    type=str,   default="nemo-multimodal-moe-muon")
    parser.add_argument("--enable_wandb",     action="store_true", default=True)
    parser.add_argument("--data_dir",         type=str,   default="./data")
    parser.add_argument("--cache_dir",        type=str,   default="./pixmo_cache")       # NEW

    parser.add_argument("--num_layers",           type=int,   default=4)
    parser.add_argument("--hidden_size",          type=int,   default=192)
    parser.add_argument("--num_attention_heads",  type=int,   default=8)
    parser.add_argument("--ffn_hidden_size",      type=int,   default=3072)
    parser.add_argument("--num_moe_experts",      type=int,   default=8)
    parser.add_argument("--seq_length",           type=int,   default=1024)
    parser.add_argument("--text_seq_len",         type=int,   default=256)               # FIXED: was 6

    parser.add_argument("--max_steps",            type=int,   default=100)
    parser.add_argument("--global_batch_size",    type=int,   default=16)

    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--adam_lr",      type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--use_amp",    action="store_true", default=True)
    parser.add_argument("--log_every",  type=int,   default=10)
    parser.add_argument("--pixmo_max_samples", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=2)


    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)
    train(args)
