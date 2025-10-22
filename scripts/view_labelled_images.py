# vis_yolo_opencv.py
import cv2
import numpy as np
from pathlib import Path

# -------- USER CONFIG ----------
IMAGES_DIR = Path("./Frames")
LABELS_DIR = Path("./dataset_split/inferred_labels")  # where your .txt files live
OUT_DIR = Path("./vis")                 # where to save annotated images
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = 'Court'  # e.g. ["person","car","cat"] or None to just show class ids
THICKNESS = 2
FONT_SCALE = 0.5
SHOW_WINDOW = True  # True to open cv2.imshow for each image (press any key to continue)
# -------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_yolo_txt(txt_path: Path):
    """Return list of (cls:int, x_center, y_center, w, h) as floats.
       If file missing or empty -> return empty list."""
    if not txt_path.exists():
        return []
    text = txt_path.read_text().strip()
    if text == "":
        return []
    labels = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        labels.append((cls, x, y, w, h))
    return labels

def yolo_to_xyxy(label, img_w, img_h):
    """Convert normalized yolo xywh -> pixel x1,y1,x2,y2 (int)"""
    cls, x_center, y_center, w, h = label
    xc = x_center * img_w
    yc = y_center * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(xc - bw/2.0))
    y1 = int(round(yc - bh/2.0))
    x2 = int(round(xc + bw/2.0))
    y2 = int(round(yc + bh/2.0))
    # clamp to image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w-1, x2)
    y2 = min(img_h-1, y2)
    return cls, x1, y1, x2, y2

def draw_boxes_on_image(img_path: Path, labels):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not read image {img_path}")
        return None
    h, w = img.shape[:2]
    for label in labels:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(label, w, h)
        color = (0, 255, 0)  # BGR; green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)
        # label text
        if CLASS_NAMES and 0 <= cls < len(CLASS_NAMES):
            text = CLASS_NAMES[cls]
        else:
            text = str(cls)
        # put filled box behind text for readability
        ((text_w, text_h), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(img, (x1, y1 - int(text_h*1.5)), (x1 + text_w, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), 1, cv2.LINE_AA)
    return img

def main():
    files = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not files:
        print("No images found.")
        return
    for p in files:
        txt = LABELS_DIR / f"{p.stem}.txt"
        labels = read_yolo_txt(txt)  # list of (cls,x,y,w,h)
        img_annot = draw_boxes_on_image(p, labels)
        if img_annot is None:
            continue
        out_p = OUT_DIR / p.name
        cv2.imwrite(str(out_p), img_annot)
        print("Saved:", out_p)
        if SHOW_WINDOW:
            cv2.imshow("vis", img_annot)
            cv2.waitKey(0)
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
