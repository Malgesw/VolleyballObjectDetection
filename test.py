import shutil
import os
from ultralytics import YOLO
from train import preprocess

name_preprocess = "threshold_and_lines"
train_epochs = 100
model_version = f"{train_epochs}_epochs_{name_preprocess}"

with open(f"params_{name_preprocess}.txt", "r") as file:
    params = file.read()

model = YOLO(
    f"./runs_{model_version}{params}/detect/yolo_train_{name_preprocess}{params}/weights/best.pt"
)
name_preprocess = "threshold_and_lines"
preprocess("test_frames", name_preprocess, params, test=True)
model.predict(source="test_frames" + name_preprocess + params, save=True, imgsz=640)
shutil.rmtree("test_frames" + name_preprocess + params)
if os.path.exists(f"./runs_{model_version}{params}/detect/predict_test_{name_preprocess}{params}"):
    shutil.rmtree(
        f"./runs_{model_version}{params}/detect/predict_test_{name_preprocess}{params}")
shutil.move("./runs/detect/predict/", f"./runs_{model_version}{params}/detect")
os.rename(
    f"./runs_{model_version}{params}/detect/predict",
    f"./runs_{model_version}{params}/detect/predict_test_{name_preprocess}{params}",
)
shutil.rmtree("./runs")
