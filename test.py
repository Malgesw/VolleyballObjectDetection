from ultralytics import YOLO

model = YOLO("./runs/detect/yolo_train/weights/best.pt")

model.predict(source="./test_cropped_frames/", save=True, imgsz=640)
