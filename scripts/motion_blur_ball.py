import os
import cv2
import numpy as np
import random
import shutil
import yaml

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_dataset = os.path.join(base_dir, "datasets", "dataset_ball_aug")
output_dataset = os.path.join(base_dir, "datasets", "dataset_ball_blurred")
blur_fraction = 0.5  
motion_blur_degree = 5  

def apply_realistic_motion_blur(image, degree=5, angle=0):
    kernel = np.zeros((degree, degree))
    kernel[int((degree - 1) / 2), :] = np.linspace(1, 0, degree)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / np.sum(kernel)
    blurred = cv2.filter2D(image, -1, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)

def random_contrast_brightness(img):
    alpha = random.uniform(0.85, 1.25) 
    beta = random.randint(-25, 25)      
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def process_split(split_name):
    split_dir = os.path.join(input_dataset, split_name)
    img_dir = os.path.join(split_dir, "images")
    lbl_dir = os.path.join(split_dir, "labels")

    out_split_dir = os.path.join(output_dataset, split_name)
    out_img_dir = os.path.join(out_split_dir, "images")
    out_lbl_dir = os.path.join(out_split_dir, "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
    random.shuffle(image_files)

    for img_name in image_files:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        shutil.copy(img_path, os.path.join(out_img_dir, img_name))
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(out_lbl_dir, os.path.basename(lbl_path)))

    n_blur = int(len(image_files) * blur_fraction)
    blur_candidates = random.sample(image_files, n_blur)

    for img_name in blur_candidates:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Immagine non valida: {img_path}")
            continue

        degree = random.randint(3, 7)
        angle = random.choice([0, 5, -5, 10, -10])
        blurred = apply_realistic_motion_blur(img, degree=degree, angle=angle)
        if random.random() < 0.5:
            blurred = random_contrast_brightness(blurred)

        blur_name = os.path.splitext(img_name)[0] + "_blur.jpg"
        cv2.imwrite(os.path.join(out_img_dir, blur_name), blurred)

        if os.path.exists(lbl_path):
            lbl_copy_path = os.path.join(out_lbl_dir, os.path.splitext(blur_name)[0] + ".txt")
            shutil.copy(lbl_path, lbl_copy_path)

    print(f"✅ {split_name}: {len(image_files)} originali + {n_blur} blur = {len(image_files) + n_blur} totali")

for split in ["train", "valid"]:
    process_split(split)

data_yaml = {
    "train": os.path.join(output_dataset, "train", "images").replace("\\", "/"),
    "val": os.path.join(output_dataset, "valid", "images").replace("\\", "/"),
    "nc": 1,
    "names": ["ball"]
}

yaml_path = os.path.join(output_dataset, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\n📄 File data.yaml creato in:\n{yaml_path}")
print("✨ Dataset con motion blur realistico (50% extra immagini) generato con successo!")
