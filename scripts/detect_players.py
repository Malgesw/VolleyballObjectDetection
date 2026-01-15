import math
import time
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from collections import deque

FIELD_DETECT_INTERVAL = 5.0
PEOPLE_DETECT_INTERVAL = 0.1
BALL_DETECT_INTERVAL = 0.01
FIELD_MODEL_PATH = "../runs/yolo_train_dataset_with_test_frames_and_augs/weights/best.pt"
PEOPLE_MODEL_PATH = "../models/yolov8m.pt"
BALL_MODEL_PATH = "../runs/yolo_train_dataset_ball_blurred/weights/best.pt"
CONF_THRESHOLD = 0.3
CONF_BALL = 0.3

OUTPUT_W, OUTPUT_H = 1920, 1080
ZOOM_MIN = 1.0
ZOOM_MAX = 3.0
ZOOM_SPEED = 1.0
ZOOM_TRIGGER_FRAC = 0.09  # fraction of frame height; if ball within this distance to a player, trigger zoom
ZOOM_OUT_DELAY = 1.0
BALL_DISTANCE_TOL = 400
current_zoom = 1.0
target_zoom = 1.0
zoom_center = None


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
        conf = float(b.conf.cpu().numpy()[0])  # estrai il primo elemento
        cls = int(b.cls.cpu().numpy()[0])  # estrai il primo elemento
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
        conf = b["conf"]
        if conf > best_conf:
            best_conf = conf
            best = b
    return best


def point_in_bbox(px, py, bbox):
    x1, y1, x2, y2 = bbox
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def get_bbox_coords(b):
    xy = b["xyxy"]
    if isinstance(xy[0], (list, tuple, np.ndarray)):
        x1, y1, x2, y2 = xy[0]
    else:
        x1, y1, x2, y2 = xy
    return float(x1), float(y1), float(x2), float(y2)


def update_zoom(current, target, dt, speed=ZOOM_SPEED):
    if dt <= 0:
        return current
    max_step = speed * dt
    diff = target - current
    if abs(diff) <= max_step:
        return target
    return current + max_step * (1 if diff > 0 else -1)


def compute_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def zoom_frame_and_get_mapping(frame, center, zoom, output_size=(OUTPUT_W, OUTPUT_H)):
    orig_h, orig_w = frame.shape[0], frame.shape[1]
    out_w, out_h = output_size
    target_ar = out_w / out_h

    base_crop_w = max(1, int(orig_w / zoom))
    base_crop_h = max(1, int(orig_h / zoom))

    if (base_crop_w / base_crop_h) > target_ar:
        crop_h = base_crop_h
        crop_w = max(1, int(crop_h * target_ar))
    else:
        crop_w = base_crop_w
        crop_h = max(1, int(crop_w / target_ar))

    cx, cy = int(center[0]), int(center[1])
    crop_x1 = cx - crop_w // 2
    crop_y1 = cy - crop_h // 2

    crop_x1 = max(0, min(crop_x1, orig_w - crop_w))
    crop_y1 = max(0, min(crop_y1, orig_h - crop_h))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.shape[0] <= 0 or crop.shape[1] <= 0:
        scale = out_w / orig_w
        zoomed = cv2.resize(frame, (out_w, out_h))
        return zoomed, 0, 0, scale

    zoomed = cv2.resize(crop, (out_w, out_h))
    scale = out_w / crop_w
    return zoomed, crop_x1, crop_y1, scale


def map_bbox_to_output(bbox, crop_x1, crop_y1, scale):
    x1, y1, x2, y2 = bbox
    nx1 = int((x1 - crop_x1) * scale)
    ny1 = int((y1 - crop_y1) * scale)
    nx2 = int((x2 - crop_x1) * scale)
    ny2 = int((y2 - crop_y1) * scale)
    return nx1, ny1, nx2, ny2


def main(args):
    global current_zoom, target_zoom, zoom_center

    print("Loading models...")
    field_model = YOLO(FIELD_MODEL_PATH)
    people_model = YOLO(PEOPLE_MODEL_PATH)
    ball_model = YOLO(BALL_MODEL_PATH)
    print("Models loaded.")

    INPUT_SOURCE = f"../test_clip_{args.clip_number}.mp4"

    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {INPUT_SOURCE}")

    last_field_time = -9999.0
    last_people_time = -9999.0
    last_ball_time = -9999.0
    last_zoom_in_time = -9999.0

    last_field_bbox = None
    last_ball_bbox = None

    last_players = []
    last_frame_time = time.time()

    chosen_center = (0, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or frame read failed.")
            break

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        # ---------------- FIELD DETECTION ----------------
        if now - last_field_time >= FIELD_DETECT_INTERVAL:
            last_field_time = now
            results = field_model(frame, imgsz=640, conf=CONF_THRESHOLD)
            field_bboxes = extract_bboxes_from_results(results, label="field", conf_thresh=CONF_THRESHOLD)
            chosen = choose_best_bbox(field_bboxes)
            if chosen:
                last_field_bbox = list(chosen["xyxy"][0]) if isinstance(chosen["xyxy"][0],
                                                                        (list, np.ndarray)) else list(chosen["xyxy"])
                print(f"Field detected, bbox={last_field_bbox}, conf={chosen['conf']:.2f}")
            else:
                print(f"Field not found on this run;")

        # ---------------- PEOPLE DETECTION ----------------
        if now - last_people_time >= PEOPLE_DETECT_INTERVAL:
            last_people_time = now
            results = people_model(frame, imgsz=640, conf=CONF_THRESHOLD, classes=[0], verbose=False)
            people_boxes = extract_bboxes_from_results(results, label="person", conf_thresh=CONF_THRESHOLD)
            last_people_boxes = people_boxes

            players = []
            if last_field_bbox is not None:
                fx1, fy1, fx2, fy2 = last_field_bbox
                for b in last_people_boxes:
                    x1, y1, x2, y2 = map(int, b["xyxy"][0])
                    lower_left_x = x1
                    lower_left_y = y2
                    if point_in_bbox(lower_left_x, lower_left_y, (fx1, fy1, fx2, fy2)):
                        players.append(b)
            last_players = players

            # ---------------- BALL DETECTION ----------------
        if now - last_ball_time >= BALL_DETECT_INTERVAL:
            last_ball_time = now
            results = ball_model(frame, imgsz=1280, conf=CONF_BALL, verbose=False)
            ball_boxes = extract_bboxes_from_results(results, label="ball", conf_thresh=CONF_BALL)
            best_ball = choose_best_bbox(ball_boxes)
            if best_ball:
                last_ball_bbox = list(best_ball["xyxy"][0]) if isinstance(best_ball["xyxy"][0],
                                                                          (list, np.ndarray)) else list(
                    best_ball["xyxy"])
            else:
                last_ball_bbox = None

        frame_h, frame_w = frame.shape[0], frame.shape[1]
        zoom_trigger_dist = ZOOM_TRIGGER_FRAC * frame_h

        if last_ball_bbox is not None:
            bx1, by1, bx2, by2 = map(int, last_ball_bbox)
            ball_cx = (bx1 + bx2) / 2.0
            ball_cy = (by1 + by2) / 2.0
            if zoom_center is not None and np.linalg.norm(np.array(zoom_center) - np.array((ball_cx, ball_cy))) < BALL_DISTANCE_TOL:
                chosen_center = (ball_cx, ball_cy)

        zoom_should_trigger = False
        if last_ball_bbox is not None and len(last_players) > 0:
            ball_center = (ball_cx, ball_cy)
            min_dist = float("inf")
            for p in last_players:
                px1, py1, px2, py2 = get_bbox_coords(p)
                p_cx = (px1 + px2) / 2.0
                p_cy = (py1 + py2) / 2.0
                d = compute_distance(ball_center, (p_cx, p_cy))
                min_dist = min(min_dist, d)
            if min_dist <= zoom_trigger_dist:
                zoom_should_trigger = True

        if zoom_should_trigger:
            target_zoom = ZOOM_MAX
            zoom_center = chosen_center
            last_zoom_in_time = now
        else:
            if (now - last_zoom_in_time) >= ZOOM_OUT_DELAY:
                target_zoom = ZOOM_MIN
                if current_zoom <= 1.05:
                    zoom_center = (frame_w//2, frame_h//2)
            else:
                target_zoom = ZOOM_MAX
                zoom_center = chosen_center

        current_zoom = update_zoom(current_zoom, target_zoom, dt, speed=ZOOM_SPEED)
        current_zoom = max(ZOOM_MIN, min(ZOOM_MAX, current_zoom))

        # ---------------- VISUALIZATION ----------------


        zoomed, crop_x1, crop_y1, scale = zoom_frame_and_get_mapping(frame, zoom_center, current_zoom,
                                                                     output_size=(OUTPUT_W, OUTPUT_H))

        vis = zoomed.copy()

        # draw ball path
        """if len(ball_path) > 1:
            for i in range(1, len(ball_path)):
                if ball_path[i - 1] is None or ball_path[i] is None:
                    continue
                cv2.line(vis, ball_path[i - 1], ball_path[i], (0, 0, 255), 2)"""

        # draw current ball bbox
        if last_ball_bbox is not None:
            bx1, by1, bx2, by2 = map(int, last_ball_bbox)
            nbx1, nby1, nbx2, nby2 = map_bbox_to_output((bx1, by1, bx2, by2), crop_x1, crop_y1, scale)
            cv2.rectangle(vis, (nbx1, nby1), (nbx2, nby2), (0, 0, 255), 2)
            cv2.putText(vis, "BALL", (nbx1, max(0, nby1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # draw field bbox if available
        if last_field_bbox is not None:
            fx1, fy1, fx2, fy2 = map(int, last_field_bbox)
            nfx1, nfy1, nfx2, nfy2 = map_bbox_to_output((fx1, fy1, fx2, fy2), crop_x1, crop_y1, scale)
            cv2.rectangle(vis, (nfx1, nfy1), (nfx2, nfy2), (0, 200, 0), 3)
            cv2.putText(vis, "Field", (nfx1, max(0, nfy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        # draw players (players on field)
        for b in last_players:
            x1, y1, x2, y2 = map(int, get_bbox_coords(b))
            nx1, ny1, nx2, ny2 = map_bbox_to_output((x1, y1, x2, y2), crop_x1, crop_y1, scale)
            cv2.rectangle(vis, (nx1, ny1), (nx2, ny2), (255, 0, 0), 3)
            cv2.putText(vis, f"PLAYER", (nx1, ny2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # draw people
        """for b in last_people_boxes:
            x1, y1, x2, y2 = map(int, b["xyxy"][0])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 2)
            cv2.putText(vis, f"person", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)"""


        cv2.imshow("Volleyball Player Detection", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_number",
        type=int,
        default=1,
        help="Number of the video clip to test (values = 1 or 2)",
    )
    args = parser.parse_args()
    main(args)
