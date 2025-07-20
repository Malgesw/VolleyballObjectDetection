import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO


def preprocess(source_dir, name, test=False):
    src = source_dir
    dst = source_dir + name
    shutil.copytree(src, dst)

    input_dir = os.path.join(dst, "train", "images")
    output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)
    if test:
        input_dir = source_dir

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
                    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(os.path.join(output_dir, filename), thresh)

        case _:
            print(f"No preprocessing defined for: '{name}'")


def main():
    num_epochs = 100
    model = YOLO("yolov8n.pt")
    name_preprocess = "threshold"
    preprocess("dataset", name_preprocess)
    data_path = "dataset" + name_preprocess + "/data.yaml"
    model.train(
        data=data_path,
        epochs=num_epochs,
        batch=10,
        imgsz=640,
        device=0,
        name="yolo_train_" + name_preprocess,
    )
    shutil.rmtree("dataset" + name_preprocess)


if __name__ == "__main__":
    main()
