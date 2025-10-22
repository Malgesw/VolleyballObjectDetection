from ultralytics import YOLO
import torch
import os

def main():
    # Dataset già pronto (croppato + label corrette)
    dataset_path = "dataset_ball_aug"
    data_path = os.path.join(dataset_path, "data.yaml")

    # Parametri di training
    num_epochs = 50
    lr0 = 1e-4
    batch_size = 8
    weight_decay = 0.0003

    params_dict = {
        "lr0": lr0,
        "weight_decay": weight_decay,
        "batch_size": batch_size
    }
    params = "".join([f"_{k}={v}" for k, v in params_dict.items()])

    # Se hai GPU usa CUDA
    device = 0 if torch.cuda.is_available() else 'cpu'

    # Carica modello pre-addestrato (più grande di "n" se puoi)
    model = YOLO("yolov8s.pt")  # puoi provare anche "yolov8m.pt" se hai VRAM

    # Training
    model.train(
        data=data_path,
        epochs=num_epochs,
        batch=batch_size,
        imgsz=640,
        device=device,
        name="yolo_train_ball" + params,
        lr0=lr0,
        weight_decay=weight_decay,
        augment=True,   # attiva tutte le augm. di default
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        patience=20     # early stopping
    )

    print("✅ Training completato!")


if __name__ == "__main__":
    main()
