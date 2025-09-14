import shutil
import os
from ultralytics import YOLO
from train import preprocess

name_preprocess = "augmented_threshold"
train_epochs = 100
model_version = f"{train_epochs}_epochs_{name_preprocess}"

with open(f"params_{name_preprocess}.txt", "r") as file:
    params = file.read()

#params = "" # uncomment if runs does not have any params

model = YOLO(
    f"./runs_{model_version}{params}/detect/yolo_train_{name_preprocess}{params}/weights/best.pt"
)
name_preprocess = "threshold_white_orange"
preprocess("test_frames", name_preprocess, params, test=True)
results = model.predict(source="test_frames" + name_preprocess + params, save=True, imgsz=640)
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

with open(f"predictions_{name_preprocess}{params}.txt", "w") as file:
    for r in results:
        boxes = r.boxes  # bounding boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())      # class index
            conf = float(box.conf[0].item())     # confidence
            xyxy = box.xyxy[0].tolist()          # [x1, y1, x2, y2]
            
            file.write(f"class: {cls_id}, conf: {conf:.4f}, box: {xyxy}\n")
