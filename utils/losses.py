import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
class GANLoss(nn.Module):
    """
    Standard GAN Loss using BCEWithLogitsLoss.
    Modified to support Soft Labels (Label Smoothing) for stability.
    """
    def __init__(self):
        super().__init__()
        # We use BCEWithLogitsLoss because it is more numerically stable than Sigmoid + BCE
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target_val):
        # Handle cases where target_val is passed as True/False or a specific float (0.9)
        if isinstance(target_val, bool):
            target_val = 1.0 if target_val else 0.0

        # Create a target tensor filled with the target_val (e.g., 0.9)
        target = torch.full_like(pred, target_val)
        return self.loss(pred, target)
class L1Loss(nn.Module):
    """
    Pixel-wise L1 Loss to ensure anatomical accuracy in X-rays.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)
class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG16 features to preserve medical textures.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG16 weights
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # We extract features up to the 16th layer (Block 3, Conv 3)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()

        # Freeze VGG parameters since we aren't training the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()
    def forward(self, fake, real):
        # VGG expects 3-channel input [B, 3, H, W]
        # Since your X-rays are repeated in the dataset, we just verify the shape
        if fake.shape[1] == 1:
            fake = fake.repeat(1, 3, 1, 1)
            real = real.repeat(1, 3, 1, 1)

        # Move to the same device as the input
        self.feature_extractor = self.feature_extractor.to(fake.device)

        fake_features = self.feature_extractor(fake)
        real_features = self.feature_extractor(real)

        return self.criterion(fake_features, real_features)