import os
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MyTrainer_LRWarmup10(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Default settings
        self.initial_lr = 3e-4
        self.warmup_epochs = 10
        self.use_warmup = True
        self.freeze_encoder = False
        self.freeze_epochs = 5

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    momentum=0.99, nesterov=True, weight_decay=3e-5)

        if self.use_warmup:
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return float(epoch + 1) / float(self.warmup_epochs)
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def on_train_start(self):
        super().on_train_start()
        if self.freeze_encoder:
            for name, param in self.network.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False
            print(f"Encoder frozen for {self.freeze_epochs} epochs")

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.freeze_encoder and self.current_epoch == self.freeze_epochs:
            for name, param in self.network.named_parameters():
                param.requires_grad = True
            print("Defrosted encoder.")
