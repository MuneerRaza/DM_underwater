import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super(CombinedLoss, self).__init__()
        self.huber_loss = nn.HuberLoss()  # Huber Loss
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')  # KL Divergence Loss
        self.alpha = alpha  # Weight for Huber Loss
        self.beta = beta    # Weight for KL Divergence Loss

    def forward(self, input, target):
        # Compute Huber Loss
        huber = self.huber_loss(input, target)
        
        # Convert input to probabilities for KL Divergence
        input_distribution = nn.functional.softmax(input, dim=1)
        target_distribution = nn.functional.softmax(target, dim=1)

        # Compute KL Divergence Loss
        kl_div = self.kl_div_loss(
            nn.functional.log_softmax(input_distribution, dim=1), 
            target_distribution
        )

        # Combine the losses
        return self.alpha * huber + self.beta * kl_div
