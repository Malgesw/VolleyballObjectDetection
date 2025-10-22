from ultralytics import YOLO
from config import TestConfig
from preprocess_utils import prepare_test_frames
import shutil, os

def main():
    config = TestConfig()

    # Prepare test frames (copy + optional preprocessing + optional TTA)
    prepare_test_frames(
        input_dir=str(config.test_frames_dir),
        output_dir=str(config.temp_test_dir),
        preprocess_name=config.preprocess_name,
        augment=config.test_time_augmentation
    )

    # Load YOLO model
    model = YOLO(str(config.model_weights))

    os.makedirs("../predictions",exist_ok=True)

    # Predict
    results = model.predict(source=str(config.temp_test_dir), save=True, imgsz=config.imgsz)

    # Move prediction folder to organized prediction folder
    predict_folder = f"../predictions/yolo_predict_{config.dataset_name}"
    detect_path = os.path.join("runs", "detect", "predict")
    if os.path.exists(predict_folder):
        shutil.rmtree(predict_folder)
    shutil.move(detect_path, predict_folder)
    shutil.rmtree("runs")

    # Save predictions TXT
    if config.save_predictions_txt:
        with open(f"../predictions/predictions_{config.dataset_name}.txt", "w") as f:
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    f.write(f"class: {cls_id}, conf: {conf:.4f}, box: {xyxy}\n")

if __name__ == "__main__":
    main()
