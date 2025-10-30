import os
import subprocess
import argparse

def extract_frames(video_folder, output_folder, fps=10):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(video_folder):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(video_folder, filename)
            video_name = os.path.splitext(filename)[0]

            video_output_folder = os.path.join(output_folder, video_name)
            os.makedirs(video_output_folder, exist_ok=True)

            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"fps={fps}",
                os.path.join(video_output_folder, f"{video_name}_%04d.jpg")
            ]

            print(f"🎥 Estrazione frame da: {video_name}")
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("✅ Estrazione frame completata!")


if __name__ == "__main__":
    VOD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DEFAULT_OUTPUT = os.path.join(VOD_ROOT, "frames", "ball_Frames")

    parser = argparse.ArgumentParser(description="Estrai frame da video usando ffmpeg")
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=10)

    args = parser.parse_args()
    extract_frames(args.video_folder, args.output_folder, args.fps)
