#!/bin/bash
mkdir -p converted_videos
for f in ./Videos/*.webm; do
  fname=$(basename "$f" .webm)
  ffmpeg -i "$f" -c:v libx264 -crf 23 -preset medium -c:a aac "converted_videos/${fname}.mp4"
done
