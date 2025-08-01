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
        output_dir = dst

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
                    lower_white = np.array([0, 0, 200])
                    upper_white = np.array([180, 40, 255])
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
                    
        case _:
            print(f"No preprocessing defined for: '{name}'")


def main():
    num_epochs = 100
    model = YOLO("yolov8n.pt")
    name_preprocess = "threshold_and_lines"
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
    os.rename("./runs", f"./runs_{num_epochs}_epochs_{name_preprocess}")


if __name__ == "__main__":
    main()
