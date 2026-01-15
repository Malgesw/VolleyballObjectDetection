# scripts/preprocess_utils.py
import shutil
from pathlib import Path
import os
import cv2
import numpy as np

# ---------------------- Utility Functions ----------------------
def get_label_dir_from_images_dir(img_dir: str):
    """
    Convert images folder path to label folder path.
    Assumes YOLO dataset structure (train/valid/test with images/labels).
    """
    return img_dir.replace("images", "labels")

# ---------------------- Classical preprocessing ----------------------
def preprocess(source_dir: str, method: str, test: bool = False):
    """
    Apply classical preprocessing techniques to images in source_dir.
    Args:
        source_dir: path to images (can be train/valid/test images)
        method: name of the preprocessing method
        test: if True, used for test frames
    """
    dst = source_dir if test else source_dir  # No extra copy needed if test=False
    os.makedirs(dst, exist_ok=True)

    input_dir = source_dir
    for filename in os.listdir(input_dir):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)

        # ------------------ Preprocessing methods ------------------
        match method:
            case "lines":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
                if lines is not None:
                    for x1, y1, x2, y2 in lines[:, 0]:
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            case "threshold":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, img = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

            case "threshold_negative":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
                img = cv2.bitwise_not(thresh)

            case "threshold_white_orange":
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 40, 255])
                lower_orange = np.array([2, 150, 110])
                upper_orange = np.array([15, 255, 250])
                mask_white = cv2.inRange(hsv, lower_white, upper_white)
                mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
                mask = cv2.bitwise_or(mask_white, mask_orange)
                img = cv2.bitwise_and(img, img, mask=mask)

            case "threshold_and_lines":
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower_white = np.array([0, 0, 170])
                upper_white = np.array([180, 70, 255])
                lower_orange = np.array([2, 150, 110])
                upper_orange = np.array([15, 255, 250])
                mask_white = cv2.inRange(hsv, lower_white, upper_white)
                mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
                mask = cv2.bitwise_or(mask_white, mask_orange)
                img_masked = cv2.bitwise_and(img, img, mask=mask)
                gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=30)
                if lines is not None:
                    for x1, y1, x2, y2 in lines[:, 0]:
                        cv2.line(img_masked, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img = img_masked

            case _:
                print(f"[WARNING] No preprocessing defined for method '{method}'")

        # save processed image
        cv2.imwrite(os.path.join(dst, filename), img)


def save_params(params: dict, name: str):
    import json
    from pathlib import Path
    params_file = Path(f"../params/params_{name}.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=4)
    print(f"[INFO] Saved parameters to {params_file}")

def apply_preprocessing(dataset_path: Path, preprocess_name: str):
    """
    Apply classical preprocessing on train and valid images.
    """
    if not preprocess_name:
        return
    print(f"[INFO] Applying '{preprocess_name}' preprocessing on dataset '{dataset_path.name}'")
    preprocess(str(dataset_path / "train" / "images"), preprocess_name, test=False)
    preprocess(str(dataset_path / "valid" / "images"), preprocess_name, test=False)

def random_light_aug(img):
    import cv2, numpy as np, random
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-20, 20)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.75, 1.25), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    gamma = random.uniform(0.9, 1.1)
    inv_gamma = 1.0 / gamma
    table = np.array([((i/255.0)**inv_gamma)*255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def prepare_test_frames(input_dir: str, output_dir: str, preprocess_name=None, augment=True):
    """
    Copy test frames to temp folder, optionally apply classical preprocessing
    and random lighting augmentations.
    """

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    if preprocess_name:
        preprocess(input_dir, preprocess_name, test=True)

    if augment:
        for filename in os.listdir(output_dir):
            if not (filename.endswith(".jpg") or filename.endswith(".png")):
                continue
            img_file = os.path.join(output_dir, filename)
            img = cv2.imread(img_file)
            img_aug = random_light_aug(img)
            cv2.imwrite(img_file, img_aug)
