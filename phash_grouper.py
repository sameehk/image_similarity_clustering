import os
import cv2
import numpy as np
import shutil
import imageio.v2 as imageio
import scipy.spatial.distance
from collections import defaultdict

IMAGE_DIR = "input dataset"
OUTPUT_DIR = "output folder"
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Compute perceptual hash (pHash) for better similarity detection
def phash(image, size=32):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    dct = cv2.dct(np.float32(img))  # Apply Discrete Cosine Transform (DCT)
    dct_low_freq = dct[:8, :8]  # Extract 8x8 top-left DCT coefficients
    median = np.median(dct_low_freq)
    return (dct_low_freq > median).flatten()

# Compute hashes for all images
hash_dict = {img: phash(img) for img in image_files}

# Group images based on similarity using Hamming distance
groups = []
for img, hash1 in hash_dict.items():
    placed = False
    for group in groups:
        if scipy.spatial.distance.hamming(hash1, hash_dict[group[0]]) < 0.3:  # Adjusted threshold
            group.append(img)
            placed = True
            break
    if not placed:
        groups.append([img])

# Copy images into separate folders
for i, group in enumerate(groups):
    group_folder = os.path.join(OUTPUT_DIR, f'group_{i}')
    os.makedirs(group_folder, exist_ok=True)
    for img in group:
        shutil.copy2(img, os.path.join(group_folder, os.path.basename(img)))  # Copy instead of move

print(f"Images grouped into {len(groups)} categories.")
