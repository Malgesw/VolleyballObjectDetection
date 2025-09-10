import cv2
import os
import shutil

def crop_and_fix_labels(src_img, src_lbl, dst_img, dst_lbl, crop_x, crop_y, crop_w, crop_h):
    """
    Applica crop rettangolare e aggiorna le label YOLO.
    """
    img = cv2.imread(src_img)
    if img is None:
        return False
    
    H, W, _ = img.shape

    # Crop immagine
    cropped = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    if cropped.size == 0:
        return False
    
    # Carica label YOLO
    if not os.path.exists(src_lbl):
        return False
    
    new_labels = []
    with open(src_lbl, "r") as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            # Converti da normalizzato -> pixel
            x, y, w, h = x * W, y * H, w * W, h * H
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2

            # Trasla rispetto al crop
            x1 -= crop_x
            x2 -= crop_x
            y1 -= crop_y
            y2 -= crop_y

            # Clipping alle dimensioni del crop
            x1 = max(0, min(crop_w, x1))
            x2 = max(0, min(crop_w, x2))
            y1 = max(0, min(crop_h, y1))
            y2 = max(0, min(crop_h, y2))

            # Scarta box vuoti
            if x2 <= x1 or y2 <= y1:
                continue

            # Ricalcola in formato YOLO normalizzato
            new_w = (x2 - x1) / crop_w
            new_h = (y2 - y1) / crop_h
            new_x = (x1 + x2) / 2 / crop_w
            new_y = (y1 + y2) / 2 / crop_h

            new_labels.append(f"{int(cls)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")

    # Salva immagine e label se almeno una box valida
    if new_labels:
        cv2.imwrite(dst_img, cropped)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(new_labels))
        return True
    return False


def process_dataset(src_root, dst_root, crop_x, crop_y, crop_w, crop_h):
    """
    Duplica il dataset YOLO e applica crop + aggiornamento label.
    """
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dst_root, split, "images")
        lbl_dir = os.path.join(dst_root, split, "labels")
        if not os.path.exists(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, fname.rsplit(".", 1)[0] + ".txt")

            # Applica crop + aggiorna label
            crop_and_fix_labels(
                img_path, lbl_path,
                img_path, lbl_path,
                crop_x, crop_y, crop_w, crop_h
            )

    print(f"✅ Dataset croppato e corretto salvato in: {dst_root}")


# === ESEMPIO USO ===
if __name__ == "__main__":
    source_dataset = "dataset_ball"         # dataset originale
    dest_dataset = "dataset_ball_aug"   # nuovo dataset croppato

    # Definisci il rettangolo ROI (esempio: tutta l’area del campo)
    crop_x, crop_y, crop_w, crop_h = 0, 200, 1920, 600

    process_dataset(source_dataset, dest_dataset, crop_x, crop_y, crop_w, crop_h)
