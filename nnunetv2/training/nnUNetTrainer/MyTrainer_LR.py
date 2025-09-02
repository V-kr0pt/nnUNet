from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MyTrainer_LR(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defining a lower initial learning rate
        self.initial_lr = 3e-4   