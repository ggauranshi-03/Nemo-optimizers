import math
from dataclasses import dataclass
from typing import List

import torch
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning.pytorch.optim import OptimizerModule

# Install the official Muon: pip install git+https://github.com/KellerJordan/Muon
from muon import MuonWithAuxAdam


class MuonAdamWWrapper:
    """
    Wrapper to make MuonWithAuxAdam compatible with NeMo's training infrastructure.
    Fixes state_dict compatibility with Megatron-Core checkpointing.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.mcore_optimizer = self  # Required for NeMo compatibility
        
    def step(self, closure=None):
        """Perform optimization step."""
        return self.optimizer.step(closure)
        
    def zero_grad(self, set_to_none=True):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
        
    def state_dict(self):
        """
        Get state dict and sanitize it for Megatron-Core.
        Megatron expects all state values to be Tensors (to check .shape).
        Muon stores 'step' as an int, which causes an AttributeError.
        """
        sd = self.optimizer.state_dict()
        
        if 'state' in sd:
            for param_id, param_state in sd['state'].items():
                # Iterate over keys like 'step', 'momentum_buffer', etc.
                for key, value in param_state.items():
                    # If the optimizer stored a simple int (common for 'step'),
                    # convert it to a CPU tensor so Megatron can read .shape
                    if isinstance(value, int):
                        param_state[key] = torch.tensor(value, device='cpu')
                        
        return sd
        
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.optimizer.load_state_dict(state_dict)
        
    @property
    def param_groups(self):
        """Return parameter groups."""
        return self.optimizer.param_groups


@dataclass
class MuonAdamWConfig:
    lr_muon: float = 0.02
    lr_adamw: float = 3e-4
    momentum: float = 0.95
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01


# class MuonAdamWOptimizerModule(OptimizerModule):
#     """
#     Hybrid Muon+AdamW optimizer module for NeMo using official Muon library.
#     Compatible with PyTorch 2.9+ via KellerJordan/Muon package.
#     """
    
#     def __init__(
#         self,
#         lr_muon: float = 0.02,
#         lr_adamw: float = 3e-4,
#         momentum: float = 0.95,
#         betas: tuple = (0.9, 0.95),
#         weight_decay: float = 0.01,
#         lr_scheduler=None,
#     ):
#         super().__init__(lr_scheduler=lr_scheduler)
#         self.config = MuonAdamWConfig(
#             lr_muon=lr_muon,
#             lr_adamw=lr_adamw,
#             momentum=momentum,
#             betas=betas,
#             weight_decay=weight_decay,
#         )

#     def optimizers(self, model):
#         """
#         Separate parameters and create Muon + AdamW optimizers using MuonWithAuxAdam.
        
#         Muon: 2D weight matrices (excluding embeddings and classifier head)
#         AdamW: Embeddings, biases, layer norms, classifier head, 1D parameters
#         """
#         muon_params = []
#         adamw_params = []
        
#         for name, param in model.named_parameters():
#             if not param.requires_grad:
#                 continue
                
#             # Parameter categorization
#             is_embedding = 'embed' in name.lower() or 'wte' in name.lower() or 'wpe' in name.lower()
#             is_head = 'head' in name.lower() or 'lm_head' in name.lower() or 'output_layer' in name.lower()
#             is_1d = param.ndim < 2
            
#             # Use Muon only for 2D hidden layer weights
#             if not (is_embedding or is_head or is_1d) and param.ndim >= 2:
#                 muon_params.append(param)
#             else:
#                 adamw_params.append(param)
        
#         print(f"[Optimizer] Muon (2D hidden weights): {len(muon_params)} parameters")
#         print(f"[Optimizer] AdamW (biases, norms, embeddings, heads): {len(adamw_params)} parameters")
        
#         # Create parameter groups following MuonWithAuxAdam strict requirements
#         param_groups = [
#             {
#                 "params": muon_params,
#                 "use_muon": True,
#                 "lr": self.config.lr_muon,
#                 "momentum": self.config.momentum,
#                 "weight_decay": self.config.weight_decay,
#             },
#             {
#                 "params": adamw_params,
#                 "use_muon": False,
#                 "lr": self.config.lr_adamw,
#                 "betas": self.config.betas,
#                 "weight_decay": self.config.weight_decay,
#             },
#         ]
        
#         # Create MuonWithAuxAdam optimizer
#         optimizer = MuonWithAuxAdam(param_groups)
#         optimizer.mcore_optimizer = optimizer  # For NeMo Megatron compatibility
#         return [optimizer]  # Lightning recognizes MuonWithAuxAdam as a valid torch.optim.Optimizer

class MuonAdamWOptimizerModule(OptimizerModule):
    """
    Hybrid Muon+AdamW optimizer module for NeMo using official Muon library.
    Compatible with PyTorch 2.9+ via KellerJordan/Muon package.
    Includes FIX for Megatron-Core checkpointing (step removal).
    """
    
    def __init__(
        self,
        lr_muon: float = 0.02,
        lr_adamw: float = 3e-4,
        momentum: float = 0.95,
        betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.01,
        lr_scheduler=None,
    ):
        super().__init__(lr_scheduler=lr_scheduler)
        self.config = MuonAdamWConfig(
            lr_muon=lr_muon,
            lr_adamw=lr_adamw,
            momentum=momentum,
            betas=betas,
            weight_decay=weight_decay,
        )

    def optimizers(self, model):
        """
        Separate parameters and create Muon + AdamW optimizers.
        """
        muon_params = []
        adamw_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Parameter categorization
            is_embedding = 'embed' in name.lower() or 'wte' in name.lower() or 'wpe' in name.lower()
            is_head = 'head' in name.lower() or 'lm_head' in name.lower() or 'output_layer' in name.lower()
            is_1d = param.ndim < 2
            
            # Use Muon only for 2D hidden layer weights
            if not (is_embedding or is_head or is_1d) and param.ndim >= 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        print(f"[Optimizer] Muon (2D hidden weights): {len(muon_params)} parameters")
        print(f"[Optimizer] AdamW (biases, norms, embeddings, heads): {len(adamw_params)} parameters")
        
        param_groups = [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": self.config.lr_muon,
                "momentum": self.config.momentum,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": self.config.lr_adamw,
                "betas": self.config.betas,
                "weight_decay": self.config.weight_decay,
            },
        ]
        
        # Create MuonWithAuxAdam optimizer
        optimizer = MuonWithAuxAdam(param_groups)
        
        original_state_dict = optimizer.state_dict
        
        def sanitized_state_dict():
            # Get original state
            sd = original_state_dict()
            
            # Create a shallow copy to avoid corrupting the running optimizer
            new_sd = sd.copy()
            if 'state' in new_sd:
                new_sd['state'] = new_sd['state'].copy()
                
                for param_id, param_state in list(new_sd['state'].items()):
                    # Copy the param state dict so we can modify it
                    new_param_state = param_state.copy()
                    
                    # Remove 'step' key to prevent Shape Mismatch Error in Megatron
                    if 'step' in new_param_state:
                        del new_param_state['step']
                    
                    # Ensure any other integers (unlikely) are converted, just in case
                    for key, value in new_param_state.items():
                        if isinstance(value, int):
                            new_param_state[key] = torch.tensor(value, device='cpu')
                            
                    new_sd['state'][param_id] = new_param_state
                    
            return new_sd
            
        # Overwrite the method
        optimizer.state_dict = sanitized_state_dict

        optimizer.mcore_optimizer = optimizer
        return [optimizer]

def get_muon_adamw_optimizer(
    lr_muon: float = 0.02,
    lr_adamw: float = 3e-4,
    momentum: float = 0.95,
    betas: tuple = (0.9, 0.95),
    weight_decay: float = 0.01,
):
    """
    Create hybrid Muon+AdamW optimizer for NeMo using official Muon library.
    
    Args:
        lr_muon: Learning rate for Muon (2D hidden weights). Default: 0.02
        lr_adamw: Learning rate for AdamW (biases, norms, embeddings). Default: 3e-4
        momentum: Momentum for Muon. Default: 0.95
        betas: Betas for AdamW. Default: (0.9, 0.95)
        weight_decay: Weight decay coefficient (shared). Default: 0.01
    
    Returns:
        MuonAdamWOptimizerModule configured with hybrid optimizer
    """
    return MuonAdamWOptimizerModule(
        lr_muon=lr_muon,
        lr_adamw=lr_adamw,
        momentum=momentum,
        betas=betas,
        weight_decay=weight_decay,
    )


def main():
    """Main function to run pretraining with Muon+AdamW."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NeMo GPT Pretraining with Muon+AdamW Optimizer"
    )
    parser.add_argument(
        "--name", type=str, default="gpt_muon_pretrain", help="Experiment name"
    )
    parser.add_argument("--dir", type=str, default="/results", help="Results directory")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--num_gpus_per_node", type=int, default=1, help="GPUs per node")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Attention heads")
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--global_batch_size", type=int, default=32, help="Global batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--lr_muon", type=float, default=0.02, help="Muon learning rate")
    parser.add_argument("--lr_adamw", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--momentum", type=float, default=0.95, help="Momentum for Muon")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    args = parser.parse_args()

    # Model configuration
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

    # Data module
    data = llm.MockDataModule(
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
    )

    # Strategy configuration
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ddp="megatron",
        find_unused_parameters=False,
    )

    # Hybrid Muon+AdamW optimizer using official Muon library
    optimizer = get_muon_adamw_optimizer(
        lr_muon=args.lr_muon,
        lr_adamw=args.lr_adamw,
        momentum=args.momentum,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Trainer configuration
    trainer = nl.Trainer(
        devices=args.num_gpus_per_node,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    # Logger configuration
    logger = nl.NeMoLogger(
        name=args.name,
        log_dir=args.dir,
        explicit_log_dir=f"{args.dir}/{args.name}",
        use_datetime_version=False,
        ckpt=nl.ModelCheckpoint(
            save_last=True,
            save_top_k=3,
            every_n_train_steps=100,
            monitor="reduced_train_loss",
            filename="{epoch}-{step}-{reduced_train_loss:.4f}",
        ),
    )

    # Resume configuration
    resume = nl.AutoResume(resume_if_exists=False)

    # Train the model
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        optim=optimizer,
        resume=resume,
    )


if __name__ == "__main__":
    main()
