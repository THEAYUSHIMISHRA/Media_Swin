import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from arch.generator import MediSwinGenerator
import config


def get_epoch_prediction(model, image_path, epoch, device):
    """Loads a specific epoch checkpoint and returns the restored image."""
    ckpt_path = f"checkpoints/ckpt_epoch_{epoch}.pth"
    if not os.path.exists(ckpt_path):
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['G_state_dict'] if 'G_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    img = Image.open(image_path).convert("L")
    input_tensor = transform(img).unsqueeze(0).to(device).repeat(1, 3, 1, 1)

    with torch.no_grad():
        output = model(input_tensor)

    # Denormalize [-1, 1] -> [0, 1]
    output = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0)
    return output.squeeze(0).cpu().permute(1, 2, 0).numpy()


def show_evolution(image_path, epochs):
    device = config.DEVICE
    model = MediSwinGenerator().to(device)

    n = len(epochs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for i, epoch in enumerate(epochs):
        img_np = get_epoch_prediction(model, image_path, epoch, device)
        if img_np is not None:
            axes[i].imshow(img_np)
            axes[i].set_title(f"Epoch {epoch}", fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, f"Missing\nEp {epoch}", ha='center')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("results/training_evolution_milestones.png")
    plt.show()


if __name__ == "__main__":
    # Pick one interesting COVID X-ray
    TEST_IMAGE = os.path.join(config.DATASET_PATH, "COVID", os.listdir(os.path.join(config.DATASET_PATH, "COVID"))[0])

    # Select the milestones you want to compare
    MILESTONES = [0, 5, 10, 15, 25, 29]

    show_evolution(TEST_IMAGE, MILESTONES)