import os
import shutil
from ultralytics import YOLO

name_preprocess = "ball"
num_epochs = 50  # deve essere lo stesso del train

# Leggo i parametri salvati
with open(f"params_{name_preprocess}.txt", "r") as file:
    params = file.read().strip()

# Percorso atteso
weights_path = os.path.join(
    f"runs_{num_epochs}_epochs_{name_preprocess}{params}",
    "detect",
    f"yolo_train_{name_preprocess}{params}",
    "weights",
    "best.pt"
)


# Controllo se il file esiste
if not os.path.isfile(weights_path):
    print("❌ File NON trovato! Ecco la lista dei file nella cartella detect:")
    detect_dir = os.path.dirname(weights_path)
    for root, dirs, files in os.walk(detect_dir):
        for name in files:
            print(os.path.join(root, name))
    raise FileNotFoundError(f"File best.pt non trovato in {weights_path}")

print(f"✅ Carico modello da: {weights_path}")
model = YOLO(weights_path)

# Qui fai le predizioni (ad esempio su una cartella di test)
test_source = "test_frames"
model.predict(source=test_source, save=True, imgsz=640)

# Puoi opzionalmente spostare i risultati come nel tuo script originale
if os.path.exists(f"./runs_{num_epochs}_epochs_{name_preprocess}{params}/detect/predict_test_{name_preprocess}{params}"):
    shutil.rmtree(
        f"./runs_{num_epochs}_epochs_{name_preprocess}{params}/detect/predict_test_{name_preprocess}{params}"
    )

shutil.move("./runs/detect/predict/",
            f"./runs_{num_epochs}_epochs_{name_preprocess}{params}/detect")
os.rename(
    f"./runs_{num_epochs}_epochs_{name_preprocess}{params}/detect/predict",
    f"./runs_{num_epochs}_epochs_{name_preprocess}{params}/detect/predict_test_{name_preprocess}{params}"
)
shutil.rmtree("./runs")
