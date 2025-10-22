#!/usr/bin/env python3
"""
infer_and_split_yolo_remove_empty.py

Infer YOLOv8 labels for images and split into train/valid (3/4 : 1/4).
Images that produce no detections (empty labels) will be removed from the
dataset (moved to OUT_DIR/removed by default).

Requirements:
  pip install ultralytics pillow

Edit the variables under "USER CONFIG" to set paths / thresholds / behavior.
"""

from pathlib import Path
import random
import shutil
from PIL import Image
from ultralytics import YOLO
import sys
import os

# ------------------------
# USER CONFIG (edit here)
# ------------------------
IMAGES_DIR = Path("./Frames")                 # folder containing source images
MODEL_PATH = "best.pt"                        # path to your YOLOv8 weights or model name
OUT_DIR = Path("./dataset_split")             # output root
INFERRED_LABELS_DIR = OUT_DIR / "inferred_labels"  # where we cache inferred txt files
REMOVED_DIR = OUT_DIR / "removed"             # where to move images with no detections
IMG_EXTS = ["jpg", "jpeg", "png"]             # image extensions to consider
IMG_SIZE = 640                                # inference image size (imgsz)
CONF_THRESH = 0.10                            # confidence threshold
IOU_THRESH = 0.45                             # NMS IoU threshold
DEVICE = None                                 # None for autoselect, or "cpu", "0", "cuda:0", etc.
TRAIN_RATIO = 0.75                            # fraction for train split
SEED = 1234                                   # random seed for reproducibility
MOVE_IMAGES = False                            # if True, move images into output (deletes original)
DRY_RUN = False                                # if True, don't actually copy/move files; just print actions
CREATE_EMPTY_LABELS = False                    # if True, create empty .txt when model finds no boxes
REMOVE_EMPTY_IMAGES = True                     # if True, remove images that have empty labels
PERMANENTLY_DELETE_REMOVED = False            # if True, permanently delete removed images instead of moving to REMOVED_DIR
PRINT_EVERY = 50                               # progress print frequency
# ------------------------

# ---------- helpers ----------
def find_images(src: Path, exts):
    exts = [e.lower().lstrip('.') for e in exts]
    imgs = [p for p in sorted(src.iterdir()) if p.is_file() and p.suffix.lower().lstrip('.') in exts]
    return imgs

def ensure_dirs(root: Path):
    for sub in ("train/images", "train/labels", "valid/images", "valid/labels"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    INFERRED_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if REMOVE_EMPTY_IMAGES and not PERMANENTLY_DELETE_REMOVED:
        REMOVED_DIR.mkdir(parents=True, exist_ok=True)

def write_yolo_txt(txt_path: Path, labels):
    """
    labels: list of (class_id:int, x_center:float, y_center:float, w:float, h:float) normalized in [0,1]
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if not labels:
        if CREATE_EMPTY_LABELS:
            txt_path.write_text("")  # empty file if no detections
        return
    lines = []
    for cls, x, y, w, h in labels:
        lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    txt_path.write_text("\n".join(lines))

def infer_labels_for_image(model, img_path: Path, imgsz: int, conf: float, iou: float, device: str):
    """
    Returns list of (cls, x_center_norm, y_center_norm, w_norm, h_norm)
    """
    try:
        results = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    except Exception as e:
        print(f"[ERROR] inference failed for {img_path}: {e}")
        return []
    if not results or len(results) == 0:
        return []

    r = results[0]
    labels = []

    # Preferred: normalized xywh (xywhn)
    try:
        if hasattr(r.boxes, "xywhn"):
            xywhn = r.boxes.xywhn.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else [0] * len(xywhn)
            for ci, arr in zip(cls_arr, xywhn):
                x, y, w, h = arr.tolist()
                labels.append((int(ci), float(x), float(y), float(w), float(h)))
            return labels
    except Exception:
        pass

    # Fallback: xywh (absolute) then normalize by image size
    try:
        if hasattr(r.boxes, "xywh"):
            xywh = r.boxes.xywh.cpu().numpy()  # absolute pixels: x_center,y_center,w,h
            cls_arr = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else [0] * len(xywh)
            with Image.open(img_path) as im:
                W, H = im.size
            for ci, arr in zip(cls_arr, xywh):
                x, y, w, h = arr.tolist()
                labels.append((int(ci), float(x / W), float(y / H), float(w / W), float(h / H)))
            return labels
    except Exception:
        pass

    # Last fallback: try xyxy -> convert to xywh normalized
    try:
        if hasattr(r.boxes, "xyxy"):
            xyxy = r.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
            cls_arr = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else [0] * len(xyxy)
            with Image.open(img_path) as im:
                W, H = im.size
            for ci, arr in zip(cls_arr, xyxy):
                x1, y1, x2, y2 = arr.tolist()
                w = x2 - x1
                h = y2 - y1
                x = x1 + w / 2.0
                y = y1 + h / 2.0
                labels.append((int(ci), float(x / W), float(y / H), float(w / W), float(h / H)))
            return labels
    except Exception:
        pass

    # If nothing worked, return empty
    return []

def copy_to_split(img_list, labels_src_dir: Path, out_dir: Path, split_name: str, move=False, dry_run=False):
    img_out_dir = out_dir / f"{split_name}/images"
    lbl_out_dir = out_dir / f"{split_name}/labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in img_list:
        base = img_path.stem
        img_dst = img_out_dir / img_path.name
        lbl_src = labels_src_dir / f"{base}.txt"
        lbl_dst = lbl_out_dir / f"{base}.txt"

        if dry_run:
            print(f"[DRY] {split_name} image: {img_path} -> {img_dst}")
        else:
            if move:
                shutil.move(str(img_path), str(img_dst))
            else:
                shutil.copy2(str(img_path), str(img_dst))

        if dry_run:
            print(f"[DRY] {split_name} label: {lbl_src} -> {lbl_dst}")
        else:
            if lbl_src.exists():
                shutil.copy2(str(lbl_src), str(lbl_dst))
            else:
                if CREATE_EMPTY_LABELS:
                    lbl_dst.write_text("")

# ------------------------
# Script run
# ------------------------
def main():
    # sanity checks
    if not IMAGES_DIR.exists():
        print(f"[FATAL] images folder does not exist: {IMAGES_DIR}")
        sys.exit(1)

    images = find_images(IMAGES_DIR, IMG_EXTS)
    if not images:
        print(f"[FATAL] no images found in {IMAGES_DIR} with extensions {IMG_EXTS}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_dirs(OUT_DIR)
    INFERRED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")

    total = len(images)
    print(f"Starting inference for {total} images (imgsz={IMG_SIZE}, conf={CONF_THRESH}, iou={IOU_THRESH})")

    kept_images = []   # images that have at least one detection (or we keep them by policy)
    removed_count = 0

    for i, img_path in enumerate(images, start=1):
        labels = infer_labels_for_image(model, img_path, imgsz=IMG_SIZE, conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE)
        txt_path = INFERRED_LABELS_DIR / f"{img_path.stem}.txt"
        # write label file (if there are labels or if CREATE_EMPTY_LABELS True)
        write_yolo_txt(txt_path, labels)

        # if labels empty and user requested removal -> remove the image
        if (not labels) and REMOVE_EMPTY_IMAGES:
            removed_count += 1
            if DRY_RUN:
                print(f"[DRY] would remove image with empty label: {img_path}")
            else:
                if PERMANENTLY_DELETE_REMOVED:
                    try:
                        img_path.unlink()
                        print(f"[REMOVED] deleted {img_path}")
                    except Exception as e:
                        print(f"[ERROR] could not delete {img_path}: {e}")
                else:
                    # move to removed dir
                    dest = REMOVED_DIR / img_path.name
                    try:
                        shutil.move(str(img_path), str(dest))
                        print(f"[REMOVED] moved {img_path} -> {dest}")
                    except Exception as e:
                        print(f"[ERROR] could not move {img_path} to {dest}: {e}")
            # do not add to kept_images (so it will not be split)
        else:
            # keep image for splitting
            kept_images.append(img_path)

        if (i % PRINT_EVERY == 0) or (i == total):
            print(f"Inferred {i}/{total} (removed so far: {removed_count})")

    # split kept_images into train/valid
    random.seed(SEED)
    shuffled = kept_images.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * TRAIN_RATIO))
    train_imgs = shuffled[:n_train]
    valid_imgs = shuffled[n_train:]

    print(f"Total original images: {total}")
    print(f"Kept images (>=1 detection or kept by policy): {n}")
    print(f" -> train: {len(train_imgs)}, valid: {len(valid_imgs)}")
    print("Copying images and inferred labels into train/ and valid/ ...")
    copy_to_split(train_imgs, INFERRED_LABELS_DIR, OUT_DIR, "train", move=MOVE_IMAGES, dry_run=DRY_RUN)
    copy_to_split(valid_imgs, INFERRED_LABELS_DIR, OUT_DIR, "valid", move=MOVE_IMAGES, dry_run=DRY_RUN)

    print("Done.")
    print(f"Output ready at: {OUT_DIR}")
    print("Structure:")
    print(f"  {OUT_DIR}/train/images  {OUT_DIR}/train/labels")
    print(f"  {OUT_DIR}/valid/images  {OUT_DIR}/valid/labels")
    print(f"Inferred labels cached in: {INFERRED_LABELS_DIR}")
    if REMOVE_EMPTY_IMAGES:
        if PERMANENTLY_DELETE_REMOVED:
            print(f"Images permanently deleted because label empty: {removed_count}")
        else:
            print(f"Images moved to {REMOVED_DIR} because label empty: {removed_count}")

if __name__ == "__main__":
    main()
