import torch
from nnUNet.nnunetv2.training.nnUNetTrainer.MyTrainer_LRWarmup10 import MyTrainer_LRWarmup10


class MyTrainer_LRWarmup5(MyTrainer_LRWarmup10):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Default settings
        self.initial_lr = 3e-4
        self.warmup_epochs = 5
        self.use_warmup = True
        self.freeze_encoder = False
        self.freeze_epochs = 10