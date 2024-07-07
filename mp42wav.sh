#!/bin/bash

input_directory="raw/"
output_directory="raw_wav/"

# Ensure the output directory exists
mkdir -p "$output_directory"

# Loop through all MP4 files in the input directory
for mp4_file in "$input_directory"/*.mp4; do
    # Extract the filename (without extension) from the full path
    filename=$(basename -- "$mp4_file")
    filename_without_extension="${filename%.*}"

    # Construct the output WAV file path
    output_wav="$output_directory/$filename_without_extension.wav"

    # Convert MP4 to WAV using FFmpeg
    ffmpeg -i "$mp4_file" "$output_wav"
done
