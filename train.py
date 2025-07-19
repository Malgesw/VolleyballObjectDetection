from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="./dataset/data.yaml",
    epochs=50,
    batch=10,
    imgsz=640,
    device="cpu",
    name="yolo_train",
)
