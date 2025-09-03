import torch
from nnUNet.nnunetv2.training.nnUNetTrainer.MyTrainer_LRWarmup10 import MyTrainer_LRWarmup10

class MyTrainer_LR1e4_nowarmup(MyTrainer_LRWarmup10):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Default settings
        self.initial_lr = 1e-4
        self.warmup_epochs = 0
        self.use_warmup = False
        self.freeze_encoder = False
        self.freeze_epochs = 0
