import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.xray_dataset import XRayDataset
from arch.generator import MediSwinGenerator
from arch.discriminator import PatchGANDiscriminator
from utils.losses import GANLoss, L1Loss, PerceptualLoss
import config
from tqdm import tqdm
import os


def add_instance_noise(batch, std=0.01):
    """Adds a small amount of Gaussian noise to break Discriminator overconfidence."""
    if std > 0:
        return batch + torch.randn_like(batch) * std
    return batch


def train():
    device = config.DEVICE
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Dataset & Loader
    dataset = XRayDataset(config.DATASET_PATH)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    # 2. Models
    G = MediSwinGenerator().to(device)
    D = PatchGANDiscriminator().to(device)

    # 3. Losses & Optimizers
    gan_loss = GANLoss().to(device)
    l1_loss = L1Loss().to(device)
    perc_loss = PerceptualLoss().to(device)

    # Using separate LRs from config
    opt_G = torch.optim.Adam(G.parameters(), lr=config.LR_G, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=config.LR_D, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler('cuda')

    # --- FULL RESUME LOGIC ---
    checkpoint_path = "checkpoints/latest_checkpoint.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)


        G.load_state_dict(ckpt['G_state_dict'])
        D.load_state_dict(ckpt['D_state_dict'])
        opt_G.load_state_dict(ckpt['opt_G_state_dict'])
        opt_D.load_state_dict(ckpt['opt_D_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])

        start_epoch = ckpt['epoch'] + 1
        print(f"Starting from Epoch {start_epoch}")
    else:
        print("Starting Fresh Training...")

    # 4. Training Loop
    for epoch in range(start_epoch, config.EPOCHS):
        loop = tqdm(loader)
        loop.set_description(f"Epoch [{epoch}/{config.EPOCHS}]")

        for batch in loop:
            clean = batch["clean"].to(device)
            degraded = batch["degraded"].to(device)

            # --- TRAIN DISCRIMINATOR ---
            with torch.amp.autocast('cuda'):
                fake = G(degraded)

                # Apply tiny noise to D's inputs only
                # This makes it harder for D to win instantly
                D_real = D(add_instance_noise(clean, std=0.01))
                D_fake = D(add_instance_noise(fake.detach(), std=0.01))

                # LABEL SMOOTHING: 0.9 for Real instead of 1.0
                loss_D_real = gan_loss(D_real, 0.9)
                loss_D_fake = gan_loss(D_fake, 0.0)
                loss_D = (loss_D_real + loss_D_fake) / 2

            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            scaler.step(opt_D)

            # --- TRAIN GENERATOR ---
            with torch.amp.autocast('cuda'):
                # D should NOT see noise when training the Generator
                D_fake_G = D(fake)
                loss_GAN = gan_loss(D_fake_G, 1.0)  # G still wants to hit 1.0
                loss_L1 = l1_loss(fake, clean)
                loss_P = perc_loss(fake, clean)

                # Higher weight on L1/Perceptual to keep medical accuracy
                loss_G = loss_GAN + (50 * loss_L1) + (5 * loss_P)

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler.step(opt_G)
            scaler.update()

            loop.set_postfix(G_loss=f"{loss_G.item():.4f}", D_loss=f"{loss_D.item():.4f}")

        # --- SAVE FULL CHECKPOINT ---
        checkpoint = {
            'epoch': epoch,
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'opt_G_state_dict': opt_G.state_dict(),
            'opt_D_state_dict': opt_D.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }

        torch.save(checkpoint, f"checkpoints/ckpt_epoch_{epoch}.pth")
        torch.save(checkpoint, "checkpoints/latest_checkpoint.pth")


if __name__ == '__main__':
    train()