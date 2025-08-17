import os
import cv2
import matplotlib.pyplot as plt

# === CONFIG ===
# metti qui il path della cartella immagini del dataset (es: train/images)
img_dir = r"E:\Politecnico\Image Processing & Computer Vision\VolleyballObjectDetection\dataset_ball_aug\train\images"
# metti qui il path della cartella label corrispondente
label_dir = img_dir.replace("images", "labels")

# scegli quante immagini vuoi controllare
n_check = 5

# prendi n_check immagini a caso
img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
img_files = img_files[:n_check]

for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

    # carica immagine
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.readlines()

        for label in labels:
            cls, x_center, y_center, bw, bh = map(float, label.split())
            # coordinate YOLO (relative) -> pixel
            x_center, y_center, bw, bh = x_center * w, y_center * h, bw * w, bh * h
            x1 = int(x_center - bw / 2)
            y1 = int(y_center - bh / 2)
            x2 = int(x_center + bw / 2)
            y2 = int(y_center + bh / 2)

            # disegna bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"cls {int(cls)}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    plt.imshow(img)
    plt.title(img_file)
    plt.axis("off")
    plt.show()
