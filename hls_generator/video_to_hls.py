#!/usr/bin/env python3

import os
import subprocess
import argparse
from pathlib import Path

def convert_to_hls(input_file, output_dir, segment_duration=6):
    """
    Convert MP4 video to HLS format with .m4s segments and .m3u8 playlist
    
    Args:
        input_file (str): Path to input MP4 file
        output_dir (str): Directory to store output files
        segment_duration (int): Duration of each segment in seconds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output paths
    output_playlist = os.path.join(output_dir, "playlist.m3u8")
    output_segments = os.path.join(output_dir, "segment_%03d.m4s")
    
    # FFmpeg command to convert video to HLS with fMP4 segments
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-hls_time", str(segment_duration),
        "-hls_playlist_type", "vod",
        "-hls_segment_type", "fmp4",  # Specify fMP4 segments
        "-hls_segment_filename", output_segments,
        "-f", "hls",
        output_playlist
    ]
    
    try:
        # Run FFmpeg command
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Successfully converted {input_file} to HLS format")
        print(f"Output files are in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert MP4 video to HLS format")
    parser.add_argument("input_file", help="Path to input MP4 file")
    parser.add_argument("--output-dir", default="../hls_output", help="Directory to store output files")
    parser.add_argument("--segment-duration", type=int, default=6, help="Duration of each segment in seconds")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    convert_to_hls(args.input_file, args.output_dir, args.segment_duration)

if __name__ == "__main__":
    main()