from ultralytics import YOLO
from config import TestBallConfig
import os
import shutil

def main():
    config = TestBallConfig()

    model = YOLO(str(config.model_weights))
    os.makedirs("../predictions", exist_ok=True)
    os.makedirs(config.runs_dir, exist_ok=True) 

    results = model.predict(
        source=str(config.test_frames_dir),  
        save=True,
        imgsz=config.imgsz,
        project=config.runs_dir,   
        name=f"yolo_test_{config.dataset_name}",  
        exist_ok=True
    )

    if config.save_predictions_txt:
        txt_file = f"../predictions/predictions_{config.dataset_name}.txt"
        with open(txt_file, "w") as f:
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    f.write(f"class: {cls_id}, conf: {conf:.4f}, box: {xyxy}\n")
        print(f"[INFO] Predizioni salvate in {txt_file}")

if __name__ == "__main__":
    main()
