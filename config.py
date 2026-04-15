from calendar import EPOCH

import torch

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = r"E:\Ayushi\Degradation_Aware_Media_Swin\data\raw"

# LR = 0.0002
# LR = 0.00005  # Using 5e-5 for maximum stability on 4GB VRAM
# --- TTUR (Two-Time Scale Update Rule) ---
# We make G faster and D slower to prevent Discriminator collapse
# LR_G = 0.0002
# LR_D = 0.00005
# EPOCHS = 20
# --- COOL DOWN PHASE SETTINGS ---
# We reduce LR significantly to "settle" the weights
# and prevent those G_loss spikes we saw at Epoch 19.
# LR_G = 0.00005  # Was 2e-4, now 5e-5 (4x slower)
# LR_D = 0.00001  # Was 5e-5, now 1e-5 (5x slower)
#
# # Extend the training by 5-10 epochs
# # EPOCHS = 25 # Increased from 20 to 30
# EPOCHS = 30 #INCREASED FROM 25 TO 30 WITH A HEAVY NOISE FOR FINE TUNING

# --- FINAL HYPERPARAMETERS FOR EPOCH 31-40 ---
# These are optimized to handle the 0.2x scale without crashing the gradients
LR_G = 0.00002  # 2e-5: Slightly slower than your previous 5e-5 to prevent overfitting
LR_D = 0.000005 # 5e-6: Keeps the Discriminator from becoming too aggressive
EPOCHS = 40