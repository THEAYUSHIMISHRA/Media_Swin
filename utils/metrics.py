# import torch
# import torch.nn.functional as F
# from skimage.metrics import structural_similarity as ssim
# import numpy as np
#
#
# def psnr(fake, real):
#     """
#     Computes Peak Signal-to-Noise Ratio.
#     Expects tensors in range [-1, 1] and shifts them to [0, 1] for calculation.
#     """
#     # Shift from [-1, 1] to [0, 1] to match the 1.0 max signal assumption
#     fake = (fake + 1) / 2.0
#     real = (real + 1) / 2.0
#
#     # Ensure values are clipped to [0, 1] to avoid tiny floating point errors
#     fake = torch.clamp(fake, 0, 1)
#     real = torch.clamp(real, 0, 1)
#
#     mse = F.mse_loss(fake, real)
#
#     if mse == 0:
#         return 100.0
#
#     # PSNR = 20 * log10(MAX_I / sqrt(MSE))
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))
#
#
# def ssim_metric(fake, real):
#     """
#     Computes Structural Similarity Index (SSIM).
#     Handles the 3-channel (repeated grayscale) X-ray format.
#     """
#     # Shift from [-1, 1] to [0, 1]
#     fake = (fake + 1) / 2.0
#     real = (real + 1) / 2.0
#
#     # Convert tensors to numpy arrays
#     # Squeeze out batch dim and move Channel to last dim: (C, H, W) -> (H, W, C)
#     fake_np = fake.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
#     real_np = real.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
#
#     # Since X-rays are repeated 3 times, we treat it as a multi-channel image
#     # We use win_size=7 to align with your Swin Transformer window size
#     return ssim(
#         fake_np,
#         real_np,
#         data_range=1.0,
#         channel_axis=2,
#         win_size=7
#     )
#
#
# def calculate_all_metrics(fake, real):
#     """
#     Helper to get both metrics at once during validation.
#     """
#     p = psnr(fake, real)
#     s = ssim_metric(fake, real)
#     return {"psnr": p.item(), "ssim": s}
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def psnr(fake, real):
    """
    Computes Peak Signal-to-Noise Ratio.
    Handles channel mismatches between Generator output and Ground Truth.
    """
    # 1. Shift from [-1, 1] to [0, 1] and clip
    fake = torch.clamp((fake + 1.0) / 2.0, 0, 1)
    real = torch.clamp((real + 1.0) / 2.0, 0, 1)

    # 2. Match channels: If fake is (1, 3, H, W) and real is (1, 1, H, W)
    if fake.shape[1] == 3 and real.shape[1] == 1:
        real = real.repeat(1, 3, 1, 1)
    elif fake.shape[1] == 1 and real.shape[1] == 3:
        fake = fake.repeat(1, 3, 1, 1)

    mse = F.mse_loss(fake, real)

    if mse == 0:
        return torch.tensor(100.0)

    # PSNR = 20 * log10(MAX_I / sqrt(MSE))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim_metric(fake, real):
    """
    Computes Structural Similarity Index (SSIM).
    Ensures both images are 3-channel (H, W, 3) for the comparison.
    """
    # 1. Shift from [-1, 1] to [0, 1] and clip
    fake = torch.clamp((fake + 1.0) / 2.0, 0, 1)
    real = torch.clamp((real + 1.0) / 2.0, 0, 1)

    # 2. Remove batch dimension
    fake_sq = fake.squeeze(0) # (C, H, W)
    real_sq = real.squeeze(0) # (C, H, W)

    # 3. Handle Grayscale vs 3-Channel mismatch
    # If one is 3-channel and other is 1-channel, repeat the 1-channel one
    if fake_sq.shape[0] == 3 and (len(real_sq.shape) == 2 or real_sq.shape[0] == 1):
        if len(real_sq.shape) == 2: real_sq = real_sq.unsqueeze(0)
        real_sq = real_sq.repeat(3, 1, 1)
    elif (len(fake_sq.shape) == 2 or fake_sq.shape[0] == 1) and real_sq.shape[0] == 3:
        if len(fake_sq.shape) == 2: fake_sq = fake_sq.unsqueeze(0)
        fake_sq = fake_sq.repeat(3, 1, 1)

    # 4. Convert to (H, W, C) numpy arrays for skimage
    fake_np = fake_sq.cpu().detach().numpy().transpose(1, 2, 0)
    real_np = real_sq.cpu().detach().numpy().transpose(1, 2, 0)

    # Using win_size=7 to align with Swin Transformer window size
    return ssim(
        fake_np,
        real_np,
        data_range=1.0,
        channel_axis=2,
        win_size=7
    )


def calculate_all_metrics(fake, real):
    """
    Helper to get both metrics at once during validation.
    """
    p = psnr(fake, real)
    s = ssim_metric(fake, real)
    return {"psnr": p.item(), "ssim": s}
