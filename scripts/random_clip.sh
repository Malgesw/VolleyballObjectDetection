#!/bin/bash
# Usage:
#   ./random_clip.sh input.mp4 output.mp4 clip_length
# Example:
#   ./random_clip.sh input.mp4 random_clip.mp4 10
# → Extracts a random 10-second clip

set -e

input="$1"
output="$2"
clip_length="$3"

if [ $# -lt 3 ]; then
  echo "Usage: $0 <input> <output> <clip_length_seconds>"
  exit 1
fi

# Get video duration in seconds (integer)
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input")
duration=${duration%.*}

# Ensure clip is shorter than video
if (( clip_length >= duration )); then
  echo "Error: Clip length must be shorter than video duration ($duration s)."
  exit 1
fi

# Calculate max possible start time
max_start=$((duration - clip_length))

# Pick a random start time
start_time=$((RANDOM % (max_start + 1)))

echo "🎲 Video duration: ${duration}s"
echo "🎬 Extracting ${clip_length}s clip starting at ${start_time}s..."

# Extract clip using ffmpeg
ffmpeg -ss "$start_time" -i "$input" -t "$clip_length" -c copy "$output" -y

echo "✅ Random clip saved to: $output"
