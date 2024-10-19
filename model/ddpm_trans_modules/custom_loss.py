import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import VGG19_Weights

class VGGPerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.vgg = vgg[:16].eval()  # Use the first 16 layers
        for param in self.vgg.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return F.mse_loss(x_features, y_features)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.huber_loss = nn.HuberLoss()  # Huber Loss
        self.perceptual_loss = VGGPerceptualLoss()  # Perceptual Loss
        self.alpha = alpha  # Weight for Huber Loss
        self.beta = beta    # Weight for Perceptual Loss

    def forward(self, input, target):
        # Compute Huber Loss
        huber = self.huber_loss(input, target)
        
        # Compute Perceptual Loss
        perceptual = self.perceptual_loss(input, target)

        # Combine the losses
        return self.alpha * huber + self.beta * perceptual
