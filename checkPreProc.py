from train import preprocess
import os
import shutil
import cv2
import matplotlib
import matplotlib.pyplot as plt

name_preprocess = "threshold_and_lines"
folder_path = f'./dataset{name_preprocess}'

if os.path.exists(folder_path):
    shutil.rmtree(folder_path)

folder_path = os.path.join(folder_path, 'train/images')
preprocess("dataset", name_preprocess)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
image_files = [img for img in os.listdir(folder_path) if img.lower().endswith(('jpg', 'png'))]
image_files = image_files[:10]

for i, ax in enumerate(axes.flatten()):
    if i < len(image_files):
        img_path = os.path.join(folder_path, image_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.show()


