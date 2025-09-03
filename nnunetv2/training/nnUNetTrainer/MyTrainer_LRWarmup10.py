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
        # Create optimizer
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            momentum=0.99,
            nesterov=True,
            weight_decay=3e-5
        )

        # Define scheduler
        if self.use_warmup:
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return float(epoch + 1) / float(self.warmup_epochs)
                return 1.0
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda
            )
        else:
            self.lr_scheduler = None  # No scheduler if warmup disabled

        # IMPORTANT: return nothing, nnUNet expects self.optimizer and self.lr_scheduler
        # to be set as attributes, not returned
        return

    def on_train_start(self):
        super().on_train_start()
        # Optionally freeze encoder for first epochs
        if self.freeze_encoder:
            for name, param in self.network.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False
            print(f"Encoder frozen for {self.freeze_epochs} epochs")

    def on_epoch_end(self):
        super().on_epoch_end()
        # Unfreeze encoder after freeze_epochs
        if self.freeze_encoder and self.current_epoch == self.freeze_epochs:
            for name, param in self.network.named_parameters():
                param.requires_grad = True
            print("Defrosted encoder.")