from pathlib import Path
import torch

class TrainConfig:
    dataset_name = "dataset_with_test_frames_and_augs"
    dataset_path = f"../datasets/{dataset_name}"
    data_yaml = f"{dataset_path}/data.yaml" 
    model_name = "yolov8s.pt"
    runs_dir = "../runs"
    imgsz = 1056
    num_epochs = 100
    batch_size = 10
    lr0 = 0.01
    weight_decay = 0.0005
    cls = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocess_name = None 

class TestConfig:
    dataset_name = "dataset_with_test_frames_and_augs"
    dataset_path = f"../datasets/{dataset_name}"
    test_frames_dir =  "../frames/test_frames"
    temp_test_dir = "../frames/test_frames_proc"
    runs_dir = "../runs"
    model_weights = f"{runs_dir}/yolo_train_{dataset_name}/weights/best.pt"
    imgsz = 640

    preprocess_name = None 
    test_time_augmentation = True
    save_predictions_txt = True

class TrainBallConfig:
    dataset_name = "dataset_ball_blurred" 
    dataset_path = f"../datasets/{dataset_name}"  
    data_yaml = f"{dataset_path}/data.yaml"       
    model_name = "yolov8s.pt"                    
    runs_dir = "../runs"
    imgsz = 1280                                   
    num_epochs = 100
    batch_size = 4
    save_period = 10
    lr0 = 0.01
    weight_decay = 0.0005
    cls = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resume = True
    preprocess_name = None 

class TestBallConfig:
    dataset_name = "dataset_ball_aug"   
    dataset_path = f"../datasets/{dataset_name}"
    test_frames_dir = "../frames/test_frames"          
    temp_test_dir = "../frames/test_frames_proc"      
    runs_dir = "../runs"
    model_weights = f"{runs_dir}/yolo_train_{dataset_name}/weights/best.pt"
    imgsz = 3840

    preprocess_name = None
    test_time_augmentation = True
    save_predictions_txt = True