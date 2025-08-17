import cv2
import os

# === Modifica questo con il percorso del tuo frame ===
img_path = r"E:\Politecnico\Image Processing & Computer Vision\VolleyballObjectDetection\test_frames\04302025_182540_000.jpg"

# Controllo esistenza
if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ File non trovato: {img_path}")

# Carico immagine
img = cv2.imread(img_path)
scale = 0.5  # 0.5 = metà, 0.3 = 30% dell’originale
img = cv2.resize(img, None, fx=scale, fy=scale)
clone = img.copy()

points = []

# Funzione richiamata quando clicchi
def select_point(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:  # click sinistro
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Seleziona ROI", img)

# Mostro la finestra
cv2.imshow("Seleziona ROI", img)
cv2.setMouseCallback("Seleziona ROI", select_point)

print("👉 Clicca 4 punti sul campo (in senso orario o antiorario).")
print("❌ Premi 'q' per uscire quando hai finito.")

while True:
    cv2.imshow("Seleziona ROI", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or len(points) >= 4:
        break

cv2.destroyAllWindows()

# Mostro il poligono selezionato
if len(points) == 4:
    print("\n✅ ROI definita:")
    roi_polygon = "np.array(" + str(points) + ", np.int32)"
    print(roi_polygon)
else:
    print("⚠️ Non hai selezionato 4 punti!")
