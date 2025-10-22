import cv2
import numpy as np
import random
import os

def random_light_variation(img):
    """
    Apply random hue, saturation, brightness, and exposure changes
    based on your training augmentations with slightly expanded ranges.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Random Hue shift (-20 to 20 degrees)
    hue_shift = random.uniform(-20, 20)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180

    # Random Saturation scaling (0.7–1.3)
    sat_mult = random.uniform(0.7, 1.3)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_mult, 0, 255)

    # Random Brightness scaling (0.75–1.25)
    val_mult = random.uniform(0.75, 1.25)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_mult, 0, 255)

    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Random Exposure / Gamma correction (0.9–1.1)
    gamma = random.uniform(0.9, 1.1)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    return img

# ------------------- Main Script -------------------
input_folder = "frames/test_frames/"       # Folder with original test frames
output_folder = "frames/test_frames_aug/"  # Folder to save augmented frames
num_variants = 3                     # How many random variants per frame

os.makedirs(output_folder, exist_ok=True)

# List only .jpg files in the folder
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    base_name = os.path.splitext(filename)[0]

    # Generate multiple random variants
    for i in range(num_variants):
        aug_img = random_light_variation(img)
        out_path = os.path.join(output_folder, f"{base_name}_v{i}.jpg")
        cv2.imwrite(out_path, aug_img)

print(f"Augmented frames saved in {output_folder}")
