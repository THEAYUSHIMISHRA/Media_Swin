import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from arch.generator import MediSwinGenerator
from utils.degradation import degrade_image  # IMPORT THIS
import config


def denormalize(tensor):
    return torch.clamp((tensor + 1.0) / 2.0, min=0.0, max=1.0)


def get_restored_batch(model, image_paths, device):
    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    degraded_list, restored_list, clean_list = [], [], []
    model.eval()

    for path in image_paths:
        # 1. Load Clean Image
        img_pil = Image.open(path).convert("L")
        img_np = np.array(img_pil)

        # 2. APPLY DEGRADATION (The Stress Test)
        # This creates the "Input" we want to fix
        degraded_np = degrade_image(img_np)
        degraded_pil = Image.fromarray(degraded_np)

        # 3. Convert both to tensors
        clean_tensor = transform(img_pil).unsqueeze(0).to(device)
        degraded_tensor = transform(degraded_pil).unsqueeze(0).to(device)

        # 4. Model takes the DEGRADED image as input
        input_tensor = degraded_tensor.repeat(1, 3, 1, 1)

        with torch.no_grad():
            output = model(input_tensor)

        # Store for plotting
        degraded_list.append(denormalize(degraded_tensor).squeeze(0))
        restored_list.append(denormalize(output).squeeze(0))
        clean_list.append(denormalize(clean_tensor).squeeze(0))

    return degraded_list, restored_list, clean_list


def final_visualize(checkpoint_epoch, num_samples=3):
    device = config.DEVICE
    model = MediSwinGenerator().to(device)

    ckpt_path = f"checkpoints/ckpt_epoch_{checkpoint_epoch}.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found!")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['G_state_dict'] if isinstance(checkpoint,
                                                          dict) and 'G_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    covid_dir = os.path.join(config.DATASET_PATH, "COVID")
    all_imgs = [os.path.join(covid_dir, f) for f in os.listdir(covid_dir)[:num_samples]]

    deg, res, cln = get_restored_batch(model, all_imgs, device)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    cols = ["Input (Degraded)", "Swin-GAN Restored", "Target (Clean)"]

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontweight='bold')

    for i in range(num_samples):
        # Degraded
        axes[i, 0].imshow(deg[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        # Restored
        axes[i, 1].imshow(res[i].permute(1, 2, 0).cpu().numpy())
        # Clean Target
        axes[i, 2].imshow(cln[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        for j in range(3):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    final_visualize(checkpoint_epoch=38, num_samples=3)