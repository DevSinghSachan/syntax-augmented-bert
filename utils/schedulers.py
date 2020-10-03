import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class NoamAdamSchedule(LambdaLR):
    def __init__(self, optimizer, freeze_steps, warmup_steps=16000, hidden_size=768, t_total=-1, last_epoch=-1):
        self.freeze_steps = freeze_steps
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.dim = hidden_size
        super(NoamAdamSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step <= self.freeze_steps:
            step += 1  # to void step == 0
            decay = (self.dim ** (-0.5)) * np.min([step ** (-0.5), step * (self.warmup_steps ** (-1.5))])
            return decay
        if step > self.freeze_steps:
            return 1e-5 * max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total)))


class FreezeWarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, freeze_steps, warmup_steps, t_total, last_epoch=-1):
        self.freeze_steps = freeze_steps
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(FreezeWarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step <= self.freeze_steps:
            return 0
        if step <= self.freeze_steps + self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
