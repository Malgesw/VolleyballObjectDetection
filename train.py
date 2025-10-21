import os
import shutil

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import albumentations as A
import ultralytics.data.augment as ua

class _NoAlbumentations:
    def __init__(self, *args, **kwargs):
        # keep fields Ultralytics expects
        self.p = 0.0
        self.transform = None
        self.contains_spatial = False

    def __call__(self, labels):
        # return labels unchanged; no augmentation applied
        return labels
orig = ua.Albumentations
ua.Albumentations = _NoAlbumentations
def get_label_dir_from_images_dir(images_dir):
    if os.path.sep + "images" + os.path.sep in images_dir:
        return images_dir.replace(os.path.sep + "images" + os.path.sep,
                                  os.path.sep + "labels" + os.path.sep)
    if images_dir.endswith(os.path.sep + "images") or images_dir.endswith("images"):
        return images_dir.rsplit(os.path.sep + "images", 1)[0] + os.path.sep + "labels"
    parent = os.path.dirname(images_dir)
    return os.path.join(parent, "labels")
def preprocess(source_dir, name, params="", test=False):
    src = source_dir
    dst = source_dir + name + params
    shutil.copytree(src, dst)

    input_dir = os.path.join(dst, "train", "images")
    output_dir = input_dir
    input_dir_val= os.path.join(dst, "valid","images")
    output_dir_val= input_dir_val
    os.makedirs(output_dir, exist_ok=True)
    if test:
        input_dir = source_dir
        output_dir = dst

    if test:
        print(f"Preprocess chosen: {name}")

    match name:
        case "lines":
            for filename in os.listdir(input_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    lines = cv2.HoughLinesP(
                        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
                    )
                    if lines is not None:
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(output_dir, filename), img)

        case "threshold":
            for filename in os.listdir(input_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(
                        gray, 210, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(os.path.join(output_dir, filename), thresh)

        case "threshold_negative":
            for filename in os.listdir(input_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(
                        gray, 210, 255, cv2.THRESH_BINARY)
                    thresh_neg = cv2.bitwise_not(thresh)
                    cv2.imwrite(os.path.join(output_dir, filename), thresh_neg)

        case "threshold_white_orange":
            for filename in os.listdir(input_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    lower_white = np.array([0, 0, 200])
                    upper_white = np.array([180, 40, 255])
                    lower_orange = np.array([2, 150, 110])
                    upper_orange = np.array([15, 255, 250])
                    mask_white = cv2.inRange(hsv, lower_white, upper_white)
                    mask_purple = cv2.inRange(hsv, lower_orange, upper_orange)
                    mask = cv2.bitwise_or(mask_white, mask_purple)
                    result = cv2.bitwise_and(img, img, mask=mask)
                    cv2.imwrite(os.path.join(output_dir, filename), result)

        case "threshold_opening":
            for filename in os.listdir(input_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(
                        gray, 210, 255, cv2.THRESH_BINARY)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
                    no_lines_vert = cv2.morphologyEx(
                        thresh, cv2.MORPH_OPEN, kernel)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
                    no_lines = cv2.morphologyEx(
                        no_lines_vert, cv2.MORPH_OPEN, kernel)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    no_lines = cv2.morphologyEx(
                        no_lines, cv2.MORPH_OPEN, kernel)
                    only_lines = cv2.subtract(thresh, no_lines)
                    cv2.imwrite(os.path.join(output_dir, filename), only_lines)

        case "threshold_and_lines":
            for filename in os.listdir(input_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    lower_white = np.array([0, 0, 170])
                    upper_white = np.array([180, 70, 255])
                    lower_orange = np.array([2, 150, 110])
                    upper_orange = np.array([15, 255, 250])
                    mask_white = cv2.inRange(hsv, lower_white, upper_white)
                    mask_purple = cv2.inRange(hsv, lower_orange, upper_orange)
                    mask = cv2.bitwise_or(mask_white, mask_purple)
                    img = cv2.bitwise_and(img, img, mask=mask)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    lines = cv2.HoughLinesP(
                        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=30
                    )
                    if lines is not None:
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(output_dir, filename), img)

        case "threshold_hitmiss":
            for filename in os.listdir(input_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(
                        gray, 210, 255, cv2.THRESH_BINARY)
                    # Horizontal line pattern
                    kernel_h = np.array(
                        [[0, 0], [1, 1], [0, 0]], dtype=np.uint8)

                    # Vertical line pattern
                    kernel_v = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.uint8)

                    hitmiss_h = cv2.morphologyEx(
                        thresh, cv2.MORPH_HITMISS, kernel_h)
                    hitmiss_v = cv2.morphologyEx(
                        thresh, cv2.MORPH_HITMISS, kernel_v)

                    # Combine the hits (detected lines)
                    lines = cv2.bitwise_or(hitmiss_h, hitmiss_v)
                    cv2.imwrite(os.path.join(output_dir, filename), lines)

        case "augmented_threshold_and_lines":
            img_size = 640
            input_label_dir = get_label_dir_from_images_dir(input_dir)
            output_label_dir = get_label_dir_from_images_dir(output_dir)
            if not os.path.exists(input_label_dir):
                print(
                    f"WARNING: label dir '{input_label_dir}' does not exist.")
            transform = A.Compose([
                A.RandomScale(scale_limit=0.3, p=0.5),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, p=1.0),
                A.RandomCrop(height=img_size, width=img_size, p=1.0),
                A.Perspective(scale=(0.05, 0.15), p=0.5),

                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                ], p=0.5),
                A.MotionBlur(p=0.2),
                A.GaussNoise(p=0.2),

                A.Resize(img_size, img_size)
            ],
                bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels']
                )
            )
            for filename in os.listdir(input_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    base = os.path.splitext(filename)[0]
                    label_path = os.path.join(input_label_dir, base + ".txt")
                    bboxes = []
                    class_labels = []
                    if os.path.exists(label_path):
                        with open(label_path, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    continue
                                cls, x, y, w, h = parts
                                bboxes.append([float(x), float(y), float(w), float(h)])
                                class_labels.append(cls)
                    else:
                        pass
                    transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                    img = transformed['image']
                    bboxes_t = transformed.get('bboxes', [])
                    class_labels_t = transformed.get('class_labels', [])
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
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(img_masked, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(output_dir, filename), img_masked)
                    out_label_path = os.path.join(output_label_dir, base + ".txt")
                    with open(out_label_path, "w") as f:
                        for cls, (x, y, w, h) in zip(class_labels_t, bboxes_t):
                            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            input_label_dir = get_label_dir_from_images_dir(input_dir_val)
            output_label_dir = get_label_dir_from_images_dir(output_dir_val)
            for filename in os.listdir(input_dir_val):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    path = os.path.join(input_dir_val, filename)
                    img = cv2.imread(path)
                    base = os.path.splitext(filename)[0]
                    label_path = os.path.join(input_label_dir, base + ".txt")
                    bboxes = []
                    class_labels = []
                    if os.path.exists(label_path):
                        with open(label_path, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    continue
                                cls, x, y, w, h = parts
                                bboxes.append([float(x), float(y), float(w), float(h)])
                                class_labels.append(cls)
                    else:
                        pass
                    transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                    img = transformed['image']
                    bboxes_t = transformed.get('bboxes', [])
                    class_labels_t = transformed.get('class_labels', [])
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
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(img_masked, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(output_dir_val, filename), img_masked)
                    out_label_path = os.path.join(output_label_dir, base + ".txt")
                    with open(out_label_path, "w") as f:
                        for cls, (x, y, w, h) in zip(class_labels_t, bboxes_t):
                            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        case _:
            print(f"No preprocessing defined for: '{name}'")


def main():
    model = YOLO("yolov8n.pt")
    name_preprocess = "augmented_threshold_and_lines"
    preprocess("dataset", name_preprocess)
    data_path = "dataset" + name_preprocess + "/data.yaml"
    num_epochs = 100
    lr0 = None
    batch_size = 10
    weight_decay = None
    cls = None

    assert batch_size is not None

    params_dict = {
        "lr0": lr0 if lr0 is not None else 0.01, # default value for initial learning rate
        "weight_decay": weight_decay if weight_decay is not None else 0.0005, # default value for weight decay
        "batch_size": batch_size,
        "cls": cls if cls is not None else 0.5 # default value for classification learning strength
    }
    params = ""
    for name, value in params_dict.items():
        params = params + "_" + name + "=" + str(value)

    with open(f"params_{name_preprocess}.txt", "w") as file:
        file.write(params)

    device = 0 if torch.cuda.is_available() else 'cpu'

    model.train(
        data=data_path,
        epochs=num_epochs,
        batch=params_dict["batch_size"],
        imgsz=640,
        device=device,
        name="yolo_train_" + name_preprocess + params,
        lr0 = params_dict["lr0"],
        weight_decay=params_dict["weight_decay"],
        cls=params_dict["cls"],
    )
    shutil.rmtree("dataset" + name_preprocess)
    os.rename("./runs", f"./runs_{num_epochs}_epochs_{name_preprocess}{params}")


if __name__ == "__main__":
    main()
