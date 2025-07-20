import shutil
import os
from ultralytics import YOLO
from train import preprocess

model_version = "100_epochs_lines"
name_preprocess = "lines"
model = YOLO(
    f"./runs_{model_version}/detect/yolo_train_{name_preprocess}/weights/best.pt"
)
preprocess("test_frames", name_preprocess, test=True)
model.predict(source="test_frames" + name_preprocess, save=True, imgsz=640)
shutil.rmtree("test_frames" + name_preprocess)
if os.path.exists(f"./runs_{model_version}/detect/predict_test_{name_preprocess}"):
    shutil.rmtree(f"./runs_{model_version}/detect/predict_test_{name_preprocess}")
shutil.move("./runs/detect/predict/", f"./runs_{model_version}/detect")
os.rename(
    f"./runs_{model_version}/detect/predict",
    f"./runs_{model_version}/detect/predict_test_{name_preprocess}",
)
shutil.rmtree("./runs")
