# VolleyballObjectDetection

## 1. Description

This repository contains the code for the project of the Polytechnic Uniersity of Turin course *Image Processing and Computer Vision*; the project consists in the developement of object detection models (YOLO-based) to detect player, court lines and the ball in a volleyball amateur match video stream. The pipeline supports:
- robustness to lighting and color variations
- automatic zoom on the ball during actions

More details in the [project report](./ImageProcessingAndCV_report.pdf).

## 2. Project structure

- `scripts/` — inference, preprocessing and training utilities (notably `scripts/detect_players.py`).
- `datasets/` — YOLO-formatted datasets (images, labels, `data.yaml`).
- `frames/` — extracted and cropped frames for inspection.
- `models/` — bundled model weights (example: `yolov8m.pt`, `best.pt`).
- `runs/` — training run outputs (weights, CSV metrics, plots, sample predictions).
- `predictions/` — saved inference outputs on test frames.
- `params/` — experiment parameter files and prediction summaries.

## 3. Dependencies

- Python 3.10+
- Install with conda (recommended for `torch`):

```bash
conda create -n volley python=3.10 -y
conda activate volley
# install torch via conda (choose CUDA-enabled package if appropriate)
conda install pytorch torchvision -c pytorch -y
```

- Install the remaining Python packages with pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` lists the project-level packages; `torch`/`torchvision` are included there for convenience but installing via conda is recommended to ensure compatibility with CUDA.

## 4. How to visualize results

Use `scripts/detect_players.py` to open a test clip and visualize detections. Example (with the conda env active):

```bash
python scripts/detect_players.py --clip_number 1
```

- `--clip_number` selects `test_clip_<n>.mp4` at the repository root (valid values: `1` or `2`).
- The viewer window is interactive; press `q` to quit.
- Model paths are configured via constants at the top of `scripts/detect_players.py` — update them to point to different weights if needed.

For further analysis, inspect `runs/<run_name>/` (weights, `results.csv`, plots) and the `predictions/` folder for example annotated frames.

