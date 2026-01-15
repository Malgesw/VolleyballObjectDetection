from ultralytics import YOLO
from config import TrainConfig
from preprocess_utils import apply_preprocessing, save_params
import os
import shutil

def main():
    config = TrainConfig()

    # optional preprocessing
    apply_preprocessing(config.dataset_path, config.preprocess_name)

    # load YOLO model
    model = YOLO(config.model_name)

    # save training params
    params_dict = {
        "lr0": config.lr0,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "cls": config.cls,
        "epochs": config.num_epochs
    }
    save_params(params_dict, config.dataset_name)

    model.train(
        data=str(config.data_yaml),
        epochs=config.num_epochs,
        batch=params_dict["batch_size"],
        imgsz=config.imgsz,
        device=config.device,
        name=f"yolo_train_{config.dataset_name}",
        lr0=params_dict["lr0"],
        weight_decay=params_dict["weight_decay"],
        cls=params_dict["cls"]
    )

    # move YOLO runs folder to project root
    src_folder = os.path.join("runs/detect", f"yolo_train_{config.dataset_name}")
    dst_folder = os.path.join(config.runs_dir, f"yolo_train_{config.dataset_name}")
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    shutil.move(src_folder, dst_folder)
    shutil.rmtree("runs/")

    print(f"[INFO] Training complete. Runs folder moved to {dst_folder}")

if __name__ == "__main__":
    main()
