from pathlib import Path
import torch

# ---------------------- TRAIN CONFIG ----------------------
class TrainConfig:
    dataset_name = "dataset_with_test_frames_and_augs"
    dataset_path = f"../datasets/{dataset_name}"
    data_yaml = f"{dataset_path}/data.yaml"  # assumes Roboflow YAML exists
    model_name = "yolov8s.pt"
    runs_dir = "../runs"
    imgsz = 1056
    num_epochs = 100
    batch_size = 10
    lr0 = 0.01
    weight_decay = 0.0005
    cls = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocess_name = None  # optional classical preprocessing

# ---------------------- TEST CONFIG ----------------------
class TestConfig:
    dataset_name = "dataset_with_test_frames_and_augs"
    dataset_path = f"../datasets/{dataset_name}"
    test_frames_dir =  "../frames/test_frames"
    temp_test_dir = "../frames/test_frames_proc"
    runs_dir = "../runs"
    model_weights = f"{runs_dir}/yolo_train_{dataset_name}/weights/best.pt"
    imgsz = 640

    preprocess_name = None  # optional classical preprocessing
    test_time_augmentation = True
    save_predictions_txt = True

class TrainBallConfig:
    dataset_name = "dataset_ball_blurred"  # Nome del dataset della palla
    dataset_path = f"../datasets/{dataset_name}"  # Percorso del dataset
    data_yaml = f"{dataset_path}/data.yaml"       # YAML di Roboflow o custom
    model_name = "yolov8s.pt"                     # Modello YOLOv8
    runs_dir = "../runs"
    imgsz = 1280                                   # Dimensione immagini (oggetti piccoli come la palla)
    num_epochs = 100
    batch_size = 4
    save_period = 10
    lr0 = 0.01
    weight_decay = 0.0005
    cls = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resume = True
    preprocess_name = None  # opzionale, preprocessing classico

# ---------------------- TEST CONFIG ----------------------
class TestBallConfig:
    dataset_name = "dataset_ball_aug"   # Nome dataset di test
    dataset_path = f"../datasets/{dataset_name}"
    test_frames_dir = "../frames/test_frames"          # Cartella con frames di test
    temp_test_dir = "../frames/test_frames_proc"      # Cartella temporanea per preprocessing
    runs_dir = "../runs"
    model_weights = f"{runs_dir}/yolo_train_{dataset_name}/weights/best.pt"
    imgsz = 3840

    preprocess_name = None
    test_time_augmentation = True
    save_predictions_txt = True