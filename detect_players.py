import time
import cv2
import numpy as np
from ultralytics import YOLO


FIELD_DETECT_INTERVAL = 60.0
PEOPLE_DETECT_INTERVAL = 0.1
FIELD_MODEL_PATH = "field_model_preprocessing.pt"
PEOPLE_MODEL_PATH = "yolov8m.pt"
CONF_THRESHOLD = 0.35
INPUT_SOURCE = "trim.mp4"

def preprocess_for_field(frame):
    img = frame.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 70, 255])
    lower_orange = np.array([2, 150, 110])
    upper_orange = np.array([15, 255, 250])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.bitwise_or(mask_white, mask_orange)
    masked = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=30)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(masked, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return masked


def extract_bboxes_from_results(results, label="player", conf_thresh=0.0):
    boxes = []
    if len(results) == 0:
        return boxes
    res = results[0]
    if not hasattr(res, "boxes") or len(res.boxes) == 0:
        return boxes

    for b in res.boxes:
        xyxy = b.xyxy.cpu().numpy().astype(float).tolist()
        conf = float(b.conf.cpu().numpy())
        cls = int(b.cls.cpu().numpy())
        if conf < conf_thresh:
            continue

        boxes.append({"xyxy": xyxy, "conf": conf, "cls": cls, "label": label})
    return boxes


def choose_best_bbox(boxes):
    if not boxes:
        return None
    best = None
    best_conf = -1
    for b in boxes:
        conf= b["conf"]
        if conf > best_conf:
            best_conf = conf
            best = b
    return best


def point_in_bbox(px, py, bbox):
    x1, y1, x2, y2 = bbox
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)

def main():
    print("Loading models...")
    field_model = YOLO(FIELD_MODEL_PATH)
    people_model = YOLO(PEOPLE_MODEL_PATH)
    print("Models loaded.")
    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {INPUT_SOURCE}")


    last_field_time = -9999.0
    last_people_time = -9999.0

    last_field_bbox = None
    last_people_boxes = []
    last_players = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or frame read failed.")
            break

        now = time.time()
        if now - last_field_time >= FIELD_DETECT_INTERVAL:
            last_field_time = now
            pre = preprocess_for_field(frame)
            results = field_model(pre, imgsz=640, conf=CONF_THRESHOLD)
            field_bboxes = extract_bboxes_from_results(results,label="field", conf_thresh=CONF_THRESHOLD)
            chosen = choose_best_bbox(field_bboxes)
            if chosen:
                last_field_bbox = chosen["xyxy"]
                print(f"Field detected, bbox={last_field_bbox}, conf={chosen['conf']:.2f}")
            else:
                print(f"Field not found on this run;")

        if now - last_people_time >= PEOPLE_DETECT_INTERVAL:
            last_people_time = now
            results = people_model(frame, imgsz=640, conf=CONF_THRESHOLD,classes=[0], verbose=False) #on base yolov8n, 0 is the id for the person class
            people_boxes = extract_bboxes_from_results(results,label="person", conf_thresh=CONF_THRESHOLD)
            last_people_boxes = people_boxes
            players = []
            if last_field_bbox is not None:
                fx1, fy1, fx2, fy2 = last_field_bbox[0]
                for b in last_people_boxes:
                    x1, y1, x2, y2 = b["xyxy"][0]
                    lower_left_x = x1
                    lower_left_y = y2
                    if point_in_bbox(lower_left_x, lower_left_y, (fx1, fy1, fx2, fy2)):
                        players.append(b)
            else:
                players = []

            last_players = players
            print(f"People detected: {len(last_people_boxes)}, players on field: {len(last_players)}")
        vis = frame.copy()

        if last_field_bbox is not None:
            x1, y1, x2, y2 = map(int, last_field_bbox[0])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.putText(vis, "Field", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        for b in last_people_boxes:
            x1, y1, x2, y2 = map(int, b["xyxy"][0])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 2)
            cv2.putText(vis, f"person", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        for b in last_players:
            x1, y1, x2, y2 = map(int, b["xyxy"][0])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(vis, f"PLAYER", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Volleyball Player Detection", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
