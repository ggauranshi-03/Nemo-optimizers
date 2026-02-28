import math
import argparse
import os
import torch
from torch.optim.optimizer import Optimizer
from nemo.lightning.pytorch.optim import OptimizerModule

class AdamWOptimizerModule(OptimizerModule):
    def __init__(self, adam_w_lr: float, weight_decay: float, lr_scheduler=None):
        super().__init__(lr_scheduler=lr_scheduler)
        self.adam_w_lr = adam_w_lr
        self.weight_decay = weight_decay

    def optimizers(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        return [torch.optim.AdamW(params, lr=self.adam_w_lr, weight_decay=self.weight_decay)]