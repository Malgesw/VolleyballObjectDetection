from ultralytics import YOLO

num_epochs = 50
model = YOLO(f"./models/yolov8n_{num_epochs}_epochs.pt")

model.train(
    data="./dataset/data.yaml",
    epochs=num_epochs,
    batch=10,
    imgsz=640,
    device="cpu",
    name="yolo_train",
)
