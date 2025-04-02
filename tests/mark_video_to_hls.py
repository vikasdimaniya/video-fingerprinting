#!/usr/bin/env python3
import logging
import os
import subprocess
import shutil
import numpy as np
import argparse
import json
from pathlib import Path
import tempfile

from offmark.embed.dwt_dct_svd_encoder import DwtDctSvdEncoder
from offmark.generator.shuffler import Shuffler
from offmark.video.embedder import Embedder
from offmark.video.frame_reader import FileDecoder
from offmark.video.frame_writer import FileEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s')

def generate_payload_for_segment(segment_number):
    """
    Generate a unique payload based on segment number
    
    Args:
        segment_number: The segment number (0-based)
        
    Returns:
        np.array: Unique binary payload for the segment
    """
    # Use binary representation of segment number, padded to 8 bits
    # For segment numbers > 255, we'll wrap around
    binary = format(segment_number % 256, '08b')
    return np.array([int(bit) for bit in binary])

def segment_video(input_file, output_pattern, segment_duration=2):
    """
    Segment video into exact duration chunks without relying on keyframes
    
    Args:
        input_file: Path to input MP4 file
        output_pattern: Output pattern for segments (e.g., segments/segment_%03d.mp4)
        segment_duration: Duration of each segment in seconds
    """
    logger.info(f"Segmenting {input_file} into {segment_duration}-second segments")
    
    # Force keyframes at each segment boundary for precise cutting
    cmd = [
        'ffmpeg', '-i', input_file, 
        '-f', 'segment', 
        '-segment_time', str(segment_duration),
        '-reset_timestamps', '1',
        '-force_key_frames', f'expr:gte(t,n_forced*{segment_duration})',
        '-c:v', 'libx264', '-preset', 'fast',
        '-c:a', 'aac',
        '-map', '0',
        output_pattern
    ]
    
    subprocess.run(cmd, check=True)
    
    return sorted(Path(os.path.dirname(output_pattern)).glob('*.mp4'))

def watermark_segment(segment_file, output_file, segment_number):
    """
    Watermark a single video segment using the mark.py logic
    
    Args:
        segment_file: Path to input segment
        output_file: Path for watermarked output
        segment_number: The segment number to determine the payload
    """
    logger.info(f"Watermarking {segment_file} to {output_file} with segment number {segment_number}")
    
    # Define watermark payload based on segment number
    payload = generate_payload_for_segment(segment_number)
    logger.info(f"Segment {segment_number} payload: {payload}")
    
    r = FileDecoder(segment_file)
    w = FileEncoder(output_file, r.width, r.height)

    # Initialize Frame Embedder
    frame_embedder = DwtDctSvdEncoder()
    capacity = frame_embedder.wm_capacity((r.height, r.width, 3))

    # Initialize Generator
    generator = Shuffler(key=0)
    wm = generator.generate_wm(payload, capacity)
    frame_embedder.read_wm(wm)

    # Start watermarking and transcoding
    video_embedder = Embedder(r, frame_embedder, w)
    video_embedder.start()
    
    return payload.tolist()  # Convert to list for JSON serialization

def extract_segment_number_from_filename(filename):
    """
    Extract segment number from a filename
    
    Args:
        filename: The filename to parse
        
    Returns:
        int: The extracted segment number
    """
    # Extract the segment number from filenames like "segment_001.mp4"
    basename = os.path.basename(filename)
    
    # Default to None if not found
    segment_number = None
    
    # Try to find a number in the filename
    parts = basename.split('_')
    for part in parts:
        if part.isdigit():
            segment_number = int(part)
            break
    
    # If that fails, try to extract it with a more permissive approach
    if segment_number is None:
        import re
        match = re.search(r'(\d+)', basename)
        if match:
            segment_number = int(match.group(1))
    
    return segment_number

def convert_segments_to_hls(segment_files, hls_output_dir):
    """
    Convert multiple MP4 segments directly to HLS with m4s segments
    
    Args:
        segment_files: List of MP4 segment files
        hls_output_dir: Directory to output HLS files
    """
    logger.info(f"Converting {len(segment_files)} segments to HLS format")
    
    # Create a temp directory for the concatenated file list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        # Create ffmpeg concat file format
        for segment in segment_files:
            f.write(f"file '{os.path.abspath(segment)}'\n")
    
    # HLS output paths
    master_playlist = os.path.join(hls_output_dir, 'master.m3u8')
    segment_pattern = os.path.join(hls_output_dir, 'segment_%03d.m4s')
    playlist = os.path.join(hls_output_dir, 'playlist.m3u8')
    
    # Convert to HLS using the concat demuxer with strict segment duration
    cmd = [
        'ffmpeg', 
        '-f', 'concat', 
        '-safe', '0', 
        '-i', concat_file,
        # Set very precise keyframe intervals to match segment boundaries
        '-force_key_frames', 'expr:gte(t,n_forced*2)',
        # Video and audio codec settings
        '-c:v', 'libx264', 
        '-x264-params', 'keyint=48:min-keyint=48',  # For 24fps, this is 2-second keyframes
        '-c:a', 'aac',
        # HLS settings
        '-f', 'hls',
        '-hls_time', '2',
        '-hls_segment_type', 'fmp4',
        '-hls_flags', 'independent_segments',
        '-hls_segment_filename', segment_pattern,
        '-hls_list_size', '0',
        '-master_pl_name', 'master.m3u8',
        # Small delta to ensure segment durations are as exact as possible
        '-segment_time_delta', '0.0001',
        playlist
    ]
    
    subprocess.run(cmd, check=True)
    
    # Clean up the temp file
    os.unlink(concat_file)
    
    return master_playlist, playlist

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Watermark a video and convert to HLS')
    parser.add_argument('input_file', help='Path to input video file')
    parser.add_argument('output_dir', help='Directory to store HLS output files')
    parser.add_argument('--segment-duration', type=float, default=2.0, 
                        help='Duration of each segment in seconds (default: 2.0)')
    parser.add_argument('--clean', action='store_true', 
                        help='Clean output directory before processing')
    args = parser.parse_args()
    
    # Create directory structure
    base_dir = args.output_dir
    segments_dir = os.path.join(base_dir, 'segments')
    marked_segments_dir = os.path.join(base_dir, 'marked_segments')
    hls_dir = os.path.join(base_dir, 'hls')
    
    # Clean directory if requested
    if args.clean and os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Create required directories
    for directory in [base_dir, segments_dir, marked_segments_dir, hls_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Segment the input video
    logger.info(f"Step 1: Segmenting video {args.input_file}")
    segments = segment_video(args.input_file, 
                            os.path.join(segments_dir, 'segment_%03d.mp4'),
                            args.segment_duration)
    logger.info(f"Created {len(segments)} segments")
    
    # Step 2: Watermark each segment with a unique pattern
    logger.info("Step 2: Watermarking segments with unique patterns")
    marked_segments = []
    segment_payloads = {}  # Keep track of which payload was used for each segment
    
    for segment in segments:
        output_file = os.path.join(marked_segments_dir, f"marked_{os.path.basename(segment)}")
        segment_number = extract_segment_number_from_filename(segment)
        payload = watermark_segment(segment, output_file, segment_number)
        marked_segments.append(output_file)
        segment_payloads[str(segment_number)] = payload  # Convert keys to strings for JSON
    
    # Step 3: Convert marked segments to HLS
    logger.info("Step 3: Converting marked segments to HLS")
    master_playlist, playlist = convert_segments_to_hls(marked_segments, hls_dir)
    
    # Save segment payloads for later verification
    payload_file = os.path.join(base_dir, 'segment_payloads.json')
    with open(payload_file, 'w') as f:
        json.dump(segment_payloads, f, indent=2)
    
    # Final output
    logger.info(f"Watermarking and HLS conversion complete")
    logger.info(f"Master playlist: {master_playlist}")
    logger.info(f"Playlist: {playlist}")
    logger.info(f"Segment payloads saved to: {payload_file}")

if __name__ == '__main__':
    main() 