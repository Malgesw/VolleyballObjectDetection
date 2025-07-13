import os
import cv2
import argparse
import numpy as np


def extract_frames(video_path, frames_dir, labels_dir, frames_per_video=10):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video: {}".format(video_path))
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("No frames found in video: {}".format(video_path))
        cap.release()
        return

    indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame {} from video: {}".format(idx, video_path))
            continue

        frame_filename = f"{video_name}_{i:03d}.jpg"
        label_filename = f"{video_name}_{i:03d}.txt"

        img_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(img_path, frame)
        open(os.path.join(labels_dir, label_filename), 'a').close()

    cap.release()


def main(args):

    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.labels_dir, exist_ok=True)

    videos = [os.path.join(args.videos_dir, f) for f in os.listdir(args.videos_dir)
              if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))]
    if not videos:
        print("No video files found in directory: {}".format(args.videos_dir))
        return

    for video in videos:
        print("Processing video: {}".format(video))
        extract_frames(video, args.frames_dir,
                       args.labels_dir, args.frames_per_video)

    print("Frames extraction completed.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument('--videos_dir', type=str, required=True,
                        help="Directory containing video files.")
    parser.add_argument('--frames_dir', type=str, required=True,
                        help="Directory to save extracted frames.")
    parser.add_argument('--labels_dir', type=str,
                        required=True, help="Directory to save labels.")
    parser.add_argument('--frames_per_video', type=int, default=10,
                        help="Number of frames to extract per video.")

    args = parser.parse_args()
    main(args)
