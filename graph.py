import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import your project modules
from arch.generator import MediSwinGenerator
import config


def evaluate_checkpoint(ckpt_name, num_samples=15):
    device = config.DEVICE
    model = MediSwinGenerator().to(device)

    ckpt_path = os.path.join("checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"Skipping {ckpt_name}: File not found.")
        return None, None

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['G_state_dict'] if 'G_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    test_path = os.path.join(config.DATASET_PATH, "Normal")
    test_files = sorted(os.listdir(test_path))[-num_samples:]

    psnrs, ssims = [], []

    with torch.no_grad():
        for file in test_files:
            img = Image.open(os.path.join(test_path, file)).convert("L")
            gt_np = np.array(img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))

            input_tensor = transform(img).unsqueeze(0).to(device).repeat(1, 3, 1, 1)
            output = model(input_tensor)

            res_np = torch.clamp((output + 1.0) / 2.0, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()
            res_np = (res_np[:, :, 0] * 255).astype(np.uint8)

            psnrs.append(psnr(gt_np, res_np))
            ssims.append(ssim(gt_np, res_np))

    return np.mean(psnrs), np.mean(ssims)


def plot_training_trend(epochs, psnr_trend, ssim_trend):
    plt.figure(figsize=(14, 5))

    # PSNR Trend
    plt.subplot(1, 2, 1)
    plt.plot(epochs, psnr_trend, marker='o', linestyle='-', color='royalblue', linewidth=2)
    plt.fill_between(epochs, psnr_trend, alpha=0.1, color='royalblue')
    plt.title('PSNR Improvement Over Epochs', fontsize=12)
    plt.xlabel('Epoch Number')
    plt.ylabel('Average PSNR (dB)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # SSIM Trend
    plt.subplot(1, 2, 2)
    plt.plot(epochs, ssim_trend, marker='s', linestyle='-', color='darkorange', linewidth=2)
    plt.fill_between(epochs, ssim_trend, alpha=0.1, color='darkorange')
    plt.title('SSIM Structural Accuracy Over Epochs', fontsize=12)
    plt.xlabel('Epoch Number')
    plt.ylabel('Average SSIM Score')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("swin_gan_trend.png")
    plt.show()


if __name__ == "__main__":
    # Define the epochs you want to compare
    checkpoint_files = ["ckpt_epoch_20.pth", "ckpt_epoch_25.pth", "ckpt_epoch_32.pth", "ckpt_epoch_39.pth"]
    epoch_numbers = [20, 25, 32, 39]

    final_psnrs = []
    final_ssims = []

    for ckpt in checkpoint_files:
        print(f"Evaluating {ckpt}...")
        p, s = evaluate_checkpoint(ckpt)
        if p is not None:
            final_psnrs.append(p)
            final_ssims.append(s)

    if final_psnrs:
        plot_training_trend(epoch_numbers[:len(final_psnrs)], final_psnrs, final_ssims)