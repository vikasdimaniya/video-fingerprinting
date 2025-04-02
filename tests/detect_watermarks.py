#!/usr/bin/env python3
import logging
import os
import subprocess
import shutil
import numpy as np
import cv2
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

from offmark.degenerator.de_shuffler import DeShuffler
from offmark.extract.dwt_dct_svd_decoder import DwtDctSvdDecoder
from offmark.video.frame_reader import FileDecoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s')

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

# Custom extractor to collect patterns from each frame
class PatternCollectorExtractor:
    def __init__(self, frame_reader, frame_extractor, degenerator):
        self.frame_reader = frame_reader
        self.frame_extractor = frame_extractor
        self.degenerator = degenerator
        self.patterns = []
        
    def start(self):
        while True:
            in_frame = self.frame_reader.read()
            if in_frame is None:
                logger.info('End of input stream')
                break

            pattern = self.__extract_pattern(in_frame)
            if pattern is not None:
                self.patterns.append(pattern)

        self.frame_reader.close()
        
        # Analyze collected patterns
        if not self.patterns:
            logger.warning("No patterns collected!")
            return None, None
            
        # Convert patterns to strings for counting
        pattern_strings = [''.join(map(str, pattern)) for pattern in self.patterns]
        counter = Counter(pattern_strings)
        
        # Find most common pattern
        most_common_pattern_str, count = counter.most_common(1)[0]
        most_common_pattern = np.array([int(bit) for bit in most_common_pattern_str])
        
        # Calculate frequency of most common pattern
        frequency = count / len(self.patterns) if self.patterns else 0
        
        return most_common_pattern, frequency
        
    def __extract_pattern(self, frame_rgb):
        wm_frame_yuv = cv2.cvtColor(frame_rgb.astype(np.float32), cv2.COLOR_BGR2YUV)
        frame_yuv = self.frame_extractor.decode(wm_frame_yuv)
        pattern = self.degenerator.degenerate(frame_yuv)
        return pattern

def detect_patterns_in_segment(marked_file, expected_payload=None, segment_number=None):
    """
    Detect patterns in each frame of a watermarked segment
    
    Args:
        marked_file: Path to watermarked segment
        expected_payload: Expected watermark payload
        segment_number: Segment number to derive expected payload if not provided
    
    Returns:
        tuple: (most_common_pattern, frequency, success)
    """
    logger.info(f"Detecting patterns in {marked_file}")
    
    # If segment_number is provided but expected_payload isn't, derive it
    if expected_payload is None and segment_number is not None:
        expected_payload = generate_payload_for_segment(segment_number)
    
    # If neither is provided, try to extract segment number from filename
    if expected_payload is None and segment_number is None:
        segment_number = extract_segment_number_from_filename(marked_file)
        if segment_number is not None:
            expected_payload = generate_payload_for_segment(segment_number)
    
    # If we still don't have an expected payload, we can't verify
    if expected_payload is None:
        logger.warning(f"No expected payload for {marked_file}, cannot verify")
        return None, 0, False
    
    r = FileDecoder(marked_file)
    degenerator = DeShuffler(key=0)
    degenerator.set_shape(expected_payload.shape if isinstance(expected_payload, np.ndarray) else np.array(expected_payload).shape)
    frame_extractor = DwtDctSvdDecoder()
    
    # Use our custom pattern collector
    pattern_collector = PatternCollectorExtractor(r, frame_extractor, degenerator)
    most_common_pattern, frequency = pattern_collector.start()
    
    if most_common_pattern is None:
        logger.warning(f"No patterns found in {marked_file}")
        return None, 0, False
    
    # Check if most common pattern matches expected payload
    if isinstance(expected_payload, np.ndarray):
        success = np.array_equal(most_common_pattern, expected_payload)
    else:
        # If expected_payload is a list (from JSON), convert to numpy array
        success = np.array_equal(most_common_pattern, np.array(expected_payload))
    
    logger.info(f"Segment {segment_number}: most common pattern: {most_common_pattern}, "
                f"expected: {expected_payload}, frequency: {frequency:.2f}, matches: {success}")
    
    return most_common_pattern, frequency, success

def load_payload_mappings(payload_file):
    """
    Load payload mappings from a JSON file
    
    Args:
        payload_file: Path to JSON file with segment payloads
        
    Returns:
        dict: Segment number to payload mapping
    """
    if not os.path.exists(payload_file):
        logger.warning(f"Payload file {payload_file} not found. Will use generated payloads instead.")
        return {}
    
    with open(payload_file, 'r') as f:
        # Load segment payloads from JSON file
        # Keys are strings, values are payload lists
        segment_payloads = json.load(f)
    
    return segment_payloads

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Detect watermarks in a leaked video file')
    parser.add_argument('input_file', help='Path to leaked MP4 video file')
    parser.add_argument('output_dir', help='Directory to store detection results')
    parser.add_argument('--payload-file', 
                       help='Path to segment_payloads.json file (if not provided, will generate payloads)')
    parser.add_argument('--segment-duration', type=float, default=2.0, 
                       help='Duration of each segment in seconds (default: 2.0)')
    parser.add_argument('--clean', action='store_true', 
                       help='Clean output directory before processing')
    args = parser.parse_args()
    
    # Create directory structure
    base_dir = args.output_dir
    segments_dir = os.path.join(base_dir, 'segments')
    
    # Clean directory if requested
    if args.clean and os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Create required directories
    for directory in [base_dir, segments_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Load segment payload mappings if available
    payload_file = args.payload_file
    segment_payloads = load_payload_mappings(payload_file) if payload_file else {}
    
    # Step 2: Segment the leaked video
    logger.info(f"Step 1: Segmenting leaked video {args.input_file}")
    segments = segment_video(args.input_file, 
                            os.path.join(segments_dir, 'segment_%03d.mp4'),
                            args.segment_duration)
    logger.info(f"Created {len(segments)} segments")
    
    # Step 3: Detect watermarks in each segment
    logger.info("Step 2: Detecting watermarks in segments")
    segment_results = []
    
    for segment in segments:
        segment_number = extract_segment_number_from_filename(segment)
        # Try to find expected payload in mappings, otherwise None
        expected_payload = segment_payloads.get(str(segment_number)) if segment_payloads else None
        
        most_common_pattern, frequency, success = detect_patterns_in_segment(
            segment, expected_payload, segment_number)
        
        segment_results.append({
            'segment': segment,
            'segment_number': segment_number,
            'pattern': most_common_pattern.tolist() if most_common_pattern is not None else None,
            'expected_pattern': expected_payload,
            'frequency': frequency,
            'success': success
        })
    
    # Step 4: Save and output results
    results_file = os.path.join(base_dir, 'detection_results.json')
    
    # Convert results to JSON-serializable format
    json_results = []
    for result in segment_results:
        json_results.append({
            'segment': os.path.basename(result['segment']),
            'segment_number': result['segment_number'],
            'detected_pattern': result['pattern'],
            'expected_pattern': result['expected_pattern'],
            'frequency': result['frequency'],
            'matches_expected': result['success']
        })
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print detailed results for segments
    print("\n===== WATERMARK DETECTION RESULTS =====")
    for result in segment_results:
        segment_num = result['segment_number']
        print(f"Segment {segment_num} ({os.path.basename(result['segment'])}):")
        print(f"  Detected pattern: {result['pattern']}")
        if result['expected_pattern']:
            print(f"  Expected pattern: {result['expected_pattern']}")
            print(f"  Matches expected: {result['success']}")
        print(f"  Pattern frequency: {result['frequency']:.2f}")
        
        # Convert to binary if available
        if result['pattern']:
            binary_str = ''.join(map(str, result['pattern']))
            decimal_value = int(binary_str, 2)
            print(f"  Binary: {binary_str} (Decimal: {decimal_value})")
        
        print("-------------------------------")
    
    # Calculate overall success rate if expected patterns are available
    if any(r['expected_pattern'] for r in segment_results):
        success_count = sum(1 for r in segment_results if r['success'])
        total_count = sum(1 for r in segment_results if r['expected_pattern'] is not None)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        print("\n===== DETECTION SUMMARY =====")
        print(f"Total segments: {len(segment_results)}")
        print(f"Segments with expected patterns: {total_count}")
        print(f"Successfully detected patterns: {success_count}")
        print(f"Success rate: {success_rate*100:.2f}%")
    
    # Ordered sequence of watermarks
    print("\n===== WATERMARK SEQUENCE =====")
    ordered_results = sorted(segment_results, key=lambda r: r['segment_number'] if r['segment_number'] is not None else float('inf'))
    
    for result in ordered_results:
        if result['pattern'] is not None:
            binary_str = ''.join(map(str, result['pattern']))
            decimal_value = int(binary_str, 2)
            print(f"Segment {result['segment_number']}: {binary_str} (Decimal: {decimal_value})")
    
    logger.info(f"Detection results saved to {results_file}")

if __name__ == '__main__':
    main() 