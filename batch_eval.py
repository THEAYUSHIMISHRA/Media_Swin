import torch
import os
import numpy as np
import pandas as pd
from arch.generator import MediSwinGenerator
from utils.metrics import psnr, ssim_metric
import config
from PIL import Image
import torchvision.transforms as T


def run_batch_evaluation(checkpoint_epoch):
    device = config.DEVICE
    model = MediSwinGenerator().to(device)

    # Load Weights
    ckpt_path = f"checkpoints/ckpt_epoch_{checkpoint_epoch}.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['G_state_dict'] if 'G_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    results = []
    test_dir = os.path.join(config.DATASET_PATH, "COVID")  # You can also add 'Normal'
    test_images = os.listdir(test_dir)[:50]  # Evaluate 50 images for a solid average

    print(f"Starting Batch Evaluation for Epoch {checkpoint_epoch}...")

    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        img = Image.open(img_path).convert("L")

        # Ground Truth
        real_tensor = transform(img).unsqueeze(0).to(device)

        # Swin Input (3-channel)
        input_tensor = real_tensor.repeat(1, 3, 1, 1)

        with torch.no_grad():
            fake_tensor = model(input_tensor)

        # Calculate Metrics (utils/metrics.py will handle denormalization)
        p_val = psnr(fake_tensor, real_tensor).item()
        s_val = ssim_metric(fake_tensor, real_tensor)

        results.append({"Image": img_name, "PSNR": p_val, "SSIM": s_val})

    # Save to CSV
    df = pd.DataFrame(results)
    avg_psnr = df["PSNR"].mean()
    avg_ssim = df["SSIM"].mean()

    df.to_csv(f"results/metrics_epoch_{checkpoint_epoch}.csv", index=False)

    print("-" * 30)
    print(f"FINAL RESULTS FOR EPOCH {checkpoint_epoch}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    run_batch_evaluation(checkpoint_epoch=39)