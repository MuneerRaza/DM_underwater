import torch.nn as nn
from model.ddpm_trans_modules.style_transfer import VGGPerceptualLoss
import core.metrics as Metrics

class CombinedLoss(nn.Module):
    def __init__(self, device, perceptual_weight=1.0, ssim_weight=1.0, l2_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss().to(device)          
        self.l2_loss = nn.MSELoss().to(device)                 
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.l2_weight = l2_weight

    def forward(self, output, target):
        # Calculate individual losses
        l2_loss = self.l2_loss(output, target)
        perceptual_loss = self.perceptual_loss(output, target)
        ssim_loss = 1 - Metrics.calculate_ssim(output, target)

        # Combine losses with respective weights
        total_loss = (self.l2_weight * l2_loss +
                      self.perceptual_weight * perceptual_loss +
                      self.ssim_weight * ssim_loss)
        return total_loss