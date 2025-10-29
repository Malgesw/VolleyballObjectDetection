from ultralytics import YOLO
from config import TrainBallConfig
from preprocess_utils import apply_preprocessing, save_params
import os
import shutil

def main():
    config = TrainBallConfig()
    print(f"[INFO] Starting training for dataset: {config.dataset_name}")
    print(f"[INFO] Using device: {config.device}")
    print(f"[INFO] Dataset path: {config.dataset_path}")

    # Optional preprocessing
    if hasattr(config, "preprocess_name") and config.preprocess_name:
        apply_preprocessing(config.dataset_path, config.preprocess_name)

    # Load YOLO model
    model = YOLO(config.model_name)

    # Save training params (optional)
    params_dict = {
        "lr0": config.lr0,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "cls": config.cls,
        "epochs": config.num_epochs
    }
    save_params(params_dict, config.dataset_name)

    # Train
    model.train(
        data=str(config.data_yaml),
        epochs=config.num_epochs,
        save_period=config.save_period,
        batch=params_dict["batch_size"],
        imgsz=config.imgsz,
        device=config.device,
        name=f"yolo_train_{config.dataset_name}",
        lr0=params_dict["lr0"],
        weight_decay=params_dict["weight_decay"],
        cls=params_dict["cls"]
    )

    # Move YOLO runs folder to project root
    src_folder = os.path.join("runs/detect", f"yolo_train_{config.dataset_name}")
    dst_folder = os.path.join(config.runs_dir, f"yolo_train_{config.dataset_name}")
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    shutil.move(src_folder, dst_folder)

    print(f"[INFO] Training complete. Runs folder saved to {dst_folder}")

if __name__ == "__main__":
    main()
