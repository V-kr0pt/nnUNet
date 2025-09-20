import torch
from torch.optim.lr_scheduler import LambdaLR
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MyTrainer_LRWarmup5(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Hiperparameters
        self.initial_lr = 3e-4 
        self.warmup_epochs = 5
        self.use_warmup = True
        self.freeze_encoder = False
        self.freeze_epochs = 10

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.initial_lr)

        total_epochs = self.num_epochs

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # warmup step: increase linear until initial_lr
                return float(epoch + 1) / float(self.warmup_epochs)
            else:
                # decay step: use PolynomialLR like standard nnUNet
                progress = (epoch - self.warmup_epochs) / float(total_epochs - self.warmup_epochs)
                return (1 - progress) ** 0.9  # the same as nnUNet

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return optimizer, scheduler
