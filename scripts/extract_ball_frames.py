import os
import subprocess

# Percorso della cartella con i video
video_folder = r"C:\Users\gabri\Downloads\test_video_4"

# Cartella di destinazione dei frame
output_folder = r"C:\Users\gabri\Documents\GitHub\VOD\frames\ball_Frames\ball_frames_4"
os.makedirs(output_folder, exist_ok=True)

# Scorri tutti i file video nella cartella
for filename in os.listdir(video_folder):
    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        video_path = os.path.join(video_folder, filename)
        video_name = os.path.splitext(filename)[0]

        # Cartella per i frame di questo video
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Comando ffmpeg per estrarre i frame a 10 fps
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", "fps=10",
            os.path.join(video_output_folder, f"{video_name}_%04d.jpg")
        ]

        # Esegui il comando
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("✅ Estrazione frame completata!")
