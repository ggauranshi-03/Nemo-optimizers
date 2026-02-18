import math
import argparse
import os
import torch
from torch.optim.optimizer import Optimizer
from lightning.pytorch import LightningDataModule
from nemo.lightning.pytorch.plugins import MegatronDataSampler


# ============================================================================ #
#              CRITICAL: Monkey Patch Megatron-Core TE Bug                    #
# ============================================================================ #
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
# from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.lightning.pytorch.optim import OptimizerModule

# Multimodal specific imports
try:
    from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import CLIPVisionTransformer
    from nemo.collections.multimodal.parts.utils import create_attention_mask
except ImportError:
    print("[WARNING] Some multimodal imports not available, using fallback")

# Dataset imports
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import io
import torchvision.transforms as transforms


# Apply the patch now that megatron is loaded
delayed_patch()


# --- REAL IMPORTS from PyTorch Lightning ---
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
# ============================================================================ #
#                     Multimodal Dataset Wrapper - IMAGE FOCUSED              #
# ============================================================================ #
class MultimodalStreamingDataset(Dataset):
    """
    Streaming image dataset from Hugging Face.
    Uses actual image datasets (CIFAR-10, ImageNet, Food101, etc.)
    """
    def __init__(self, tokenizer, seq_length=1024, split='train', max_samples=10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_samples = max_samples
        
        print(f"[DATASET] Loading image dataset (split={split})...")
        
        dataset_loaded = False
        
        # Option 1: Food101 - 101k food images with 101 categories
        if not dataset_loaded:
            try:
                print("[DATASET] Attempting to load Food101 dataset...")
                self.dataset = load_dataset(
                    "food101",
                    split=split,
                    streaming=True,
                    trust_remote_code=False
                )
                self.dataset = self.dataset.take(max_samples)
                self.dataset_list = list(self.dataset)
                self.label_key = 'label'
                self.image_key = 'image'
                # Food101 has category names
                self.categories = [
                    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
                    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
                    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
                    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
                    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
                    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
                    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
                    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
                    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
                    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
                    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
                    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
                    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
                    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
                    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
                    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
                    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
                    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
                    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
                    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
                    'waffles'
                ]
                dataset_loaded = True
                print(f"[DATASET] Successfully loaded Food101 with {len(self.dataset_list)} samples")
            except Exception as e:
                print(f"[DATASET] Failed to load Food101: {e}")
        
        # Option 2: CIFAR-10 - 60k images in 10 classes
        if not dataset_loaded:
            try:
                print("[DATASET] Attempting to load CIFAR-10 dataset...")
                self.dataset = load_dataset(
                    "cifar10",
                    split=split,
                    streaming=True,
                    trust_remote_code=False
                )
                self.dataset = self.dataset.take(max_samples)
                self.dataset_list = list(self.dataset)
                self.label_key = 'label'
                self.image_key = 'img'
                self.categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                                  'dog', 'frog', 'horse', 'ship', 'truck']
                dataset_loaded = True
                print(f"[DATASET] Successfully loaded CIFAR-10 with {len(self.dataset_list)} samples")
            except Exception as e:
                print(f"[DATASET] Failed to load CIFAR-10: {e}")
        
        # Option 3: CIFAR-100 - 60k images in 100 classes
        if not dataset_loaded:
            try:
                print("[DATASET] Attempting to load CIFAR-100 dataset...")
                self.dataset = load_dataset(
                    "cifar100",
                    split=split,
                    streaming=True,
                    trust_remote_code=False
                )
                self.dataset = self.dataset.take(max_samples)
                self.dataset_list = list(self.dataset)
                self.label_key = 'fine_label'
                self.image_key = 'img'
                self.categories = [
                    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                    'worm'
                ]
                dataset_loaded = True
                print(f"[DATASET] Successfully loaded CIFAR-100 with {len(self.dataset_list)} samples")
            except Exception as e:
                print(f"[DATASET] Failed to load CIFAR-100: {e}")
        
        # Option 4: Fashion-MNIST - 70k fashion images
        if not dataset_loaded:
            try:
                print("[DATASET] Attempting to load Fashion-MNIST dataset...")
                self.dataset = load_dataset(
                    "fashion_mnist",
                    split=split,
                    streaming=True,
                    trust_remote_code=False
                )
                self.dataset = self.dataset.take(max_samples)
                self.dataset_list = list(self.dataset)
                self.label_key = 'label'
                self.image_key = 'image'
                self.categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                dataset_loaded = True
                print(f"[DATASET] Successfully loaded Fashion-MNIST with {len(self.dataset_list)} samples")
            except Exception as e:
                print(f"[DATASET] Failed to load Fashion-MNIST: {e}")
        
        # Option 5: Oxford Pets - 7k pet images
        if not dataset_loaded:
            try:
                print("[DATASET] Attempting to load Oxford-IIIT Pets dataset...")
                self.dataset = load_dataset(
                    "timm/oxford-iiit-pet",
                    split=split,
                    streaming=True,
                    trust_remote_code=False
                )
                self.dataset = self.dataset.take(max_samples)
                self.dataset_list = list(self.dataset)
                self.label_key = 'label'
                self.image_key = 'image'
                self.categories = list(range(37))  # 37 pet breeds
                dataset_loaded = True
                print(f"[DATASET] Successfully loaded Oxford Pets with {len(self.dataset_list)} samples")
            except Exception as e:
                print(f"[DATASET] Failed to load Oxford Pets: {e}")
        
        if not dataset_loaded:
            raise RuntimeError("Could not load any image dataset")
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        print(f"[DATASET] Image dataset ready with {len(self.dataset_list)} samples")
    
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        item = self.dataset_list[idx]
        
        # Extract image
        try:
            if self.image_key in item:
                image = item[self.image_key]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image).convert('RGB')
                else:
                    image = image.convert('RGB')
            else:
                # Create dummy image if not available
                image = Image.new('RGB', (224, 224), color='white')
            
            image_tensor = self.image_transform(image)
        except Exception as e:
            print(f"[DATASET] Error processing image at idx {idx}: {e}")
            # Create dummy image on error
            image_tensor = torch.zeros(3, 224, 224)
        
        # Extract label and create text description
        try:
            label_idx = item[self.label_key]
            if isinstance(self.categories, list) and label_idx < len(self.categories):
                category_name = self.categories[label_idx]
                if isinstance(category_name, str):
                    # Create a descriptive text from the category
                    text = f"An image of {category_name.replace('_', ' ')}"
                else:
                    text = f"Image category {label_idx}"
            else:
                text = f"Image category {label_idx}"
        except Exception as e:
            print(f"[DATASET] Error extracting label at idx {idx}: {e}")
            text = "An image"
        
        # Tokenize text description
        tokens = self.tokenizer.text_to_ids(text)
        
        # Pad or truncate to seq_length
        if len(tokens) < self.seq_length:
            tokens = tokens + [self.tokenizer.eos_id] * (self.seq_length - len(tokens))
        else:
            tokens = tokens[:self.seq_length]
        
        # Create position_ids (required by NeMo GPT models)
        position_ids = torch.arange(0, self.seq_length, dtype=torch.long)
        
        # Create attention_mask - MUST BE BOOLEAN for Megatron-Core
        attention_mask = torch.ones(self.seq_length, dtype=torch.bool)
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long),
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'image': image_tensor,
            'loss_mask': torch.ones(self.seq_length, dtype=torch.float)
        }


# ============================================================================ #
#                   Custom Multimodal Data Module                              #
# ============================================================================ #
# ============================================================================ #
#                   Custom Multimodal Data Module                              #
# ============================================================================ #
# from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

class MultimodalDataModule(LightningDataModule):
    """
    LightningDataModule for multimodal training with streaming.
    Integrates MegatronDataSampler so MegatronStrategy can configure num_microbatches.
    """
    def __init__(
        self,
        tokenizer,
        seq_length: int = 1024,
        global_batch_size: int = 16,
        micro_batch_size: int = 1,
        num_workers: int = 4,
        max_samples: int = 10000,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples

        self.train_dataset = None

        # This is the critical part: MegatronDataSampler wires up
        # global_batch_size / micro_batch_size and num_microbatches.
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=None,  # keep simple for testing
        )

    def setup(self, stage: str | None = None):
        if self.train_dataset is None:
            self.train_dataset = MultimodalStreamingDataset(
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                split="train",
                max_samples=self.max_samples,
            )

    def train_dataloader(self):
        # MegatronStrategy.process_dataloader() will wrap this with data_sampler
        return DataLoader(
            self.train_dataset,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

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
#                        GPU Memory Utilization Callback                       #
# ============================================================================ #
class GPUMemoryCallback(Callback):
    """
    Logs per‑iteration GPU memory usage (GB).
    Resets peak memory stats after each batch to get per‑step max.
    """
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Reset peak memory stats for accurate per‑step measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not torch.cuda.is_available():
            return

        # Only log from rank 0 to avoid duplicates
        if trainer.global_rank == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            # Free memory is approximate
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - allocated

            metrics = {
                "memory/allocated_gb": allocated,
                "memory/reserved_gb": reserved,
                "memory/max_allocated_gb": max_allocated,
                "memory/free_gb": free,
            }
            pl_module.log_dict(metrics, on_step=True, on_epoch=False, rank_zero_only=True)

            # Optional: Print every N steps for console monitoring
            if batch_idx % 10 == 0:
                print(f"[GPU Mem] Step {trainer.global_step}: "
                      f"alloc={allocated:.2f}GB | "
                      f"reserved={reserved:.2f}GB | "
                      f"peak={max_allocated:.2f}GB")


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
#                    Multimodal MoE Model Configuration                        #
# ============================================================================ #
class MultimodalMoEConfig:
    """
    Configuration for Multimodal MoE model.
    Combines vision encoder with language MoE decoder.
    """
    def __init__(
        self,
        # Language model params
        num_layers=12,
        hidden_size=768,
        ffn_hidden_size=3072,
        num_attention_heads=12,
        seq_length=1024,
        # MoE params
        num_moe_experts=8,
        moe_router_topk=2,
        moe_aux_loss_coeff=0.01,
        # Vision params
        vision_model_type="clip",
        img_h=224,
        img_w=224,
        patch_dim=16,
        **kwargs
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.num_moe_experts = num_moe_experts
        self.moe_router_topk = moe_router_topk
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.vision_model_type = vision_model_type
        self.img_h = img_h
        self.img_w = img_w
        self.patch_dim = patch_dim
        
        # Additional params
        for key, value in kwargs.items():
            setattr(self, key, value)


# ============================================================================ #
#                                Main Function                                 #
# ============================================================================ #
def main():
    parser = argparse.ArgumentParser(description="NeMo Multimodal MoE Pretraining with AdaMuon")
    parser.add_argument("--name", type=str, default="multimodal_moe_adamuon", help="Experiment name")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="Experiments directory")
    parser.add_argument("--wandb_project", type=str, default="nemo-multimodal-moe-muon", help="WandB Project")
    parser.add_argument("--wandb_offline", action="store_true", help="Run WandB offline")
    parser.add_argument("--enable_wandb", action="store_true", default=True)
    
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus_per_node", type=int, default=2)
    
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--ffn_hidden_size", type=int, default=3072)
    parser.add_argument("--num_moe_experts", type=int, default=8)
    parser.add_argument("--moe_router_topk", type=int, default=2)
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=0.01)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--global_batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=10000, help="Max samples to load from streaming dataset")
    
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--adam_lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    # Vision params
    parser.add_argument("--img_h", type=int, default=224)
    parser.add_argument("--img_w", type=int, default=224)
    parser.add_argument("--patch_dim", type=int, default=16)
    
    args = parser.parse_args()

    exp_base_dir = os.path.join(args.exp_dir, args.name)
    checkpoint_dir = os.path.join(exp_base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[MULTIMODAL MoE CONFIG]")
    print(f"  Language Model:")
    print(f"    Layers: {args.num_layers}, Hidden: {args.hidden_size}")
    print(f"    Experts: {args.num_moe_experts}, Top-K: {args.moe_router_topk}")
    print(f"    Seq Length: {args.seq_length}")
    print(f"  Vision Model:")
    print(f"    Image Size: {args.img_h}x{args.img_w}")
    print(f"    Patch Dimension: {args.patch_dim}")
    print(f"{'='*70}\n")

    # Use standard Mixtral config with MoE (language backbone)
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
    
    # Create base MoE model (will process both text and image features)
    model = llm.MixtralModel(
        config=model_config, 
        tokenizer=tokenizer,
        optim=optimizer_arg
    )

    print(f"\n{'='*70}")
    print(f"[MODEL INITIALIZED]")
    print(f"  Architecture: Multimodal MoE (Vision + Language)")
    print(f"  MoE Experts: {args.num_moe_experts}")
    print(f"  Router Top-K: {args.moe_router_topk}")
    print(f"{'='*70}\n")

    # Create custom multimodal data module with streaming
    print("[DATA] Initializing multimodal streaming dataset...")
    data = MultimodalDataModule(
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=4,
        max_samples=args.max_samples
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ddp="megatron",
        find_unused_parameters=True,  # Set to True for multimodal
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
            LayerWiseDiagnosticCallback(),
            GPUMemoryCallback(),
        ],
        gradient_clip_val=1.0,
    )

    print(f"\n{'='*70}")
    print(f"[START] Multimodal MoE Training with AdaMuon")
    print(f"  Dataset: Streaming from Hugging Face (COCO/Conceptual Captions)")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Batch Size: {args.global_batch_size} (micro: {args.micro_batch_size})")
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
