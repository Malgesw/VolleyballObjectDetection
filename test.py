from ultralytics import YOLO

model_version = "yolov8_50_epochs"

model = YOLO(f"./runs_{model_version}/detect/yolo_train/weights/best.pt")

model.predict(source="./test_cropped_frames/", save=True, imgsz=640)
