import torch as T
import torch
import math
import warnings


class CosineWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max, eta_min=0, warmup_step=0, **kwargs):
        self.warmup_step = warmup_step
        super().__init__(optimizer, T_max - warmup_step, eta_min, *kwargs)

    # Copied from CosineAnnealingLR, but adding warmup and changing self.last_epoch to
    # self.last_epoch - self.warmup_step.
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == self.warmup_step:  # also covers the case where both are 0
            return self.base_lrs
        elif self.last_epoch < self.warmup_step:
            return [base_lr * (self.last_epoch + 1) / self.warmup_step for base_lr in self.base_lrs]
        elif (self.last_epoch - self.warmup_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.warmup_step) / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_step - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    _get_closed_form_lr = None


class RootWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max, eta_min=0, warmup_step=0, **kwargs):
        self.warmup_step = warmup_step
        super().__init__(optimizer, T_max - warmup_step, eta_min, *kwargs)

    # Copied from CosineAnnealingLR, but adding warmup and changing self.last_epoch to
    # self.last_epoch - self.warmup_step.
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        rets = self.base_lrs
        rets = [ret * min(1, (self.last_epoch + 1) / self.warmup_step) for ret in rets]
        rets = [ret / math.sqrt(max(self.last_epoch + 1, self.warmup_step)) for ret in rets]
        return rets

    _get_closed_form_lr = None


class CosineWarmup2(torch.optim.lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max, eta_min=0, warmup_step=0, **kwargs):
        self.warmup_step = warmup_step
        self.T_max = T_max
        super().__init__(optimizer, T_max - warmup_step, eta_min, *kwargs)

    # Copied from CosineAnnealingLR, but adding warmup and changing self.last_epoch to
    # self.last_epoch - self.warmup_step.
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        rets = self.base_lrs
        rets = [ret * min(1, (self.last_epoch + 1) / self.warmup_step) for ret in rets]
        progress = max(0, (self.last_epoch - self.warmup_step) / self.T_max)
        rets = [ret * max(0, 0.5 * (1 + math.cos(math.pi * (progress % 1.0)))) for ret in rets]
        return rets

    _get_closed_form_lr = None


class RootDecay(torch.optim.lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max, eta_min=0, warmup_step=0, **kwargs):
        self.warmup_step = warmup_step
        super().__init__(optimizer, T_max - warmup_step, eta_min, *kwargs)

    # Copied from CosineAnnealingLR, but adding warmup and changing self.last_epoch to
    # self.last_epoch - self.warmup_step.
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        rets = self.base_lrs
        # rets = [ret * min(1, (self.last_epoch + 1) / self.warmup_step) for ret in rets]
        rets = [ret / math.sqrt(max(self.last_epoch + 1, self.warmup_step)) for ret in rets]
        return rets

    _get_closed_form_lr = None


class JustWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max, eta_min=0, warmup_step=0, **kwargs):
        self.warmup_step = warmup_step
        super().__init__(optimizer, T_max - warmup_step, eta_min, *kwargs)

    # Copied from CosineAnnealingLR, but adding warmup and changing self.last_epoch to
    # self.last_epoch - self.warmup_step.
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        rets = self.base_lrs
        rets = [ret * min(1, (self.last_epoch + 1) / self.warmup_step) for ret in rets]
        return rets

    _get_closed_form_lr = None


def get_scheduler(config, optimizer):
    if config["scheduler"] is None:
        return None, None
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=config["scheduler_reduce_factor"],
                                                           patience=config["scheduler_patience"])
        return None, scheduler
    elif config["scheduler"].lower() == "cosine":
        scheduler = CosineWarmup(optimizer,
                                 T_max=config["training_steps"],
                                 warmup_step=config["warmup_steps"],
                                 eta_min=1e-6)
        return scheduler, None
    elif config["scheduler"].lower() == "cosine2":
        scheduler = CosineWarmup2(optimizer,
                                  T_max=config["training_steps"],
                                  warmup_step=config["warmup_steps"],
                                  eta_min=1e-6)
        return scheduler, None
    elif config["scheduler"].lower() == "rootwarmup":
        scheduler = RootWarmup(optimizer,
                               T_max=config["training_steps"],
                               warmup_step=config["warmup_steps"],
                               eta_min=1e-6)
        return scheduler, None
    elif config["scheduler"].lower() == "rootwarmupx":
        epoch_scheduler = None
        """
        JustWarmup(optimizer,
                               T_max=config["training_steps"],
                               warmup_step=config["warmup_steps"],
                               eta_min=1e-6)
        """
        scheduler = RootDecay(optimizer,
                              T_max=config["training_steps"],
                              warmup_step=config["warmup_steps"],
                              eta_min=1e-6)
        return scheduler, epoch_scheduler
    elif config["scheduler"].lower() == "justwarmup":
        scheduler = JustWarmup(optimizer,
                               T_max=config["training_steps"],
                               warmup_step=config["warmup_steps"],
                               eta_min=1e-6)
        return scheduler, None
    elif config["scheduler"].lower() == "linearwarmup":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                        max_lr=config["lr"],
                                                        pct_start=config["warmup_steps"] / config["training_steps"],
                                                        anneal_strategy="linear",
                                                        total_steps=config["training_steps"])
        return scheduler, None
