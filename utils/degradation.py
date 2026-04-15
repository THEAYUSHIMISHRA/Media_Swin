import cv2
import numpy as np
import random


def degrade_image(img):
    """
    Input: img (NumPy array, 0-255)
    Output: degraded_img (NumPy array, 0-255)

    Targeting Extreme Reconstruction (Phase 2 Training)
    """
    # Ensure input is a numpy array
    img = np.array(img)

    # 1. HEAVY BLUR (9x9 Kernel)
    # Increased sigma to 1.5 to smudge edges further
    if random.random() < 0.9:
        img = cv2.GaussianBlur(img, (9, 9), 1.5)

    # 2. STRONG GAUSSIAN NOISE (Std Dev 20)
    # Creates visible grain that the Swin-GAN must learn to denoise
    if random.random() < 0.9:
        noise = np.random.normal(0, 20, img.shape)
        img = img + noise

    # 3. EXTREME DOWNSCALING (0.2x to 0.4x)
    # 0.2x scaling means a 224px image is crushed down to ~45px
    if random.random() < 0.9:
        scale = random.uniform(0.2, 0.4)
        h, w = img.shape[:2]

        # Downscale using NEAREST to create aliasing/pixelation
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

        # Upscale back to original size (Model expects original dimensions)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    # Final Clipping to maintain valid pixel range
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


# ------------------------FINAL TILL EPOCH 30
# import cv2
# import numpy as np
# import random
#
#
#
# def degrade_image(img):
#     # 1. HEAVY BLUR (Increased from 5 to 15)
#     # This will make the image look very out-of-focus
#     if random.random() < 0.8: # Increased probability to 80%
#         img = cv2.GaussianBlur(img, (7, 7), 1.0)
#
#     # 2. STRONG NOISE (Increased from 10 to 40)
#     # This will add visible "salt and pepper" graininess
#     if random.random() < 0.8:
#         noise = np.random.normal(0, 15, img.shape)
#         img = img + noise
#
#     # 3. AGGRESSIVE DOWNSCALING (Lowered scale from 0.5 to 0.2)
#     # This simulates a very low-resolution sensor
#     if random.random() < 0.8:
#         scale = random.uniform(0.4, 0.6)
#         h, w = img.shape[:2]
#         img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
#         img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
#
#     img = np.clip(img, 0, 255)
#     return img.astype(np.uint8)

# import cv2
# import numpy as np
# import random
#
# def degrade_image(img):
#
#     if random.random() < 0.5:
#         img = cv2.GaussianBlur(img,(5,5),0)
#
#     if random.random() < 0.5:
#         noise = np.random.normal(0,10,img.shape)
#         img = img + noise
#
#     if random.random() < 0.5:
#         scale = random.uniform(0.5,0.8)
#         h,w = img.shape[:2]
#         img = cv2.resize(img,(int(w*scale),int(h*scale)))
#         img = cv2.resize(img,(w,h))
#
#     img = np.clip(img,0,255)
#
#     return img.astype(np.uint8)