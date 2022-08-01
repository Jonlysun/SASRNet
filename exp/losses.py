import torch
import torchvision
import pytorch_msssim
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
import config
sys.path.append("../")
import co.lr_scheduler as lr_scheduler

class ImageSimilarityLoss(nn.Module):
    def __init__(self, inp_scale="-11"):
        super().__init__()
        self.inp_scale = inp_scale
        self.ssim_loss = pytorch_msssim.SSIM(window_size=11)

    def forward(self, es, ta, sr_result=None, vmm_map=None, x_feat_3_vmm=None):

        loss = [0.15 * torch.abs(es - ta).mean()]
        loss.append(0.85 * .5 * (1 - self.ssim_loss(es, ta)))

        return loss

