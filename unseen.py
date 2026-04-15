import torch
import os
import cv2
import numpy as np
import random
from arch.generator import MediSwinGenerator
import config
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt


# def degrade_image(img):
#     """ Your custom aggressive degradation logic """
#     img = np.array(img)  # Ensure it's a numpy array
#
#     # 1. HEAVY BLUR
#     if random.random() < 0.8:
#         img = cv2.GaussianBlur(img, (7, 7), 1.0)
#
#     # 2. STRONG NOISE
#     if random.random() < 0.8:
#         noise = np.random.normal(0, 15, img.shape)
#         img = img + noise
#
#     # 3. AGGRESSIVE DOWNSCALING
#     if random.random() < 0.8:
#         scale = random.uniform(0.4, 0.6)
#         h, w = img.shape[:2]
#         img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
#         img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
#
#     img = np.clip(img, 0, 255)
#     return Image.fromarray(img.astype(np.uint8))
def degrade_image(img):
    img = np.array(img)

    # 1. EXTREME BLUR
    # Increased kernel to (15, 15) and sigma to 3.0 for a very "smudged" look
    if random.random() < 0.9:
        img = cv2.GaussianBlur(img, (9, 9), 1.5)

    # 2. EXTREME NOISE
    # Increased standard deviation to 40 - this will create heavy grain
    if random.random() < 0.9:
        noise = np.random.normal(0, 20, img.shape)
        img = img + noise

    # 3. EXTREME DOWNSCALING
    # Scale range dropped to 0.2 - 0.4. At 0.2, a 224px image becomes ~45px
    if random.random() < 0.9:
        scale = random.uniform(0.2, 0.4)
        h, w = img.shape[:2]
        # Using INTER_NEAREST creates those "pixelated" blocks which are hard to fix
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    img = np.clip(img, 0, 255)
    return Image.fromarray(img.astype(np.uint8))

def test_on_unseen(folder_type="Normal"):
    device = config.DEVICE
    model = MediSwinGenerator().to(device)

    # Load Weights
    checkpoint = torch.load("checkpoints/ckpt_epoch_39.pth", map_location=device)
    state_dict = checkpoint['G_state_dict'] if 'G_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Model Transform (Normalizing to [-1, 1])
    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    # File Handling
    test_path = os.path.join(config.DATASET_PATH, folder_type)
    test_files = sorted(os.listdir(test_path))
    sample_img_path = os.path.join(test_path, test_files[-1])

    # 1. Load Original PIL Image
    original_pil = Image.open(sample_img_path).convert("L")
    original_pil = original_pil.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))

    # 2. Apply your Degradation Function
    degraded_pil = degrade_image(original_pil)

    # 3. Convert Degraded to Tensor for Model
    input_tensor = transform(degraded_pil).unsqueeze(0).to(device).repeat(1, 3, 1, 1)

    with torch.no_grad():
        output = model(input_tensor)

    # Post-processing for visualization
    # Denormalize output from [-1, 1] to [0, 1]
    restored = torch.clamp((output + 1.0) / 2.0, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()

    # --- Plotting Results ---
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_pil, cmap='gray')
    plt.title(f"Original {folder_type} (GT)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(degraded_pil, cmap='gray')
    plt.title("Input (Degraded)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(restored)
    plt.title("Swin-GAN Restoration")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_on_unseen(folder_type="Normal")