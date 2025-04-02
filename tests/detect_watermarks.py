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

def generate_payload_for_segment(segment_number, copy_index=0):
    """
    Generate a unique payload based on segment number and copy index
    
    Args:
        segment_number: The segment number (0-based)
        copy_index: The copy index for this segment (0-based)
        
    Returns:
        np.array: Unique binary payload for the segment and copy
    """
    # We'll use the top 4 bits for segment number (0-15) and bottom 4 bits for copy index (0-15)
    # For segment numbers > 15, we'll wrap around
    segment_bits = format(segment_number % 16, '04b')
    copy_bits = format(copy_index % 16, '04b')
    binary = segment_bits + copy_bits
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

def decode_watermark_pattern(pattern):
    """
    Decode a watermark pattern into segment number and copy index
    
    Args:
        pattern: The watermark pattern (numpy array or list of 8 bits)
        
    Returns:
        tuple: (segment_number, copy_index)
    """
    if pattern is None:
        return None, None
        
    # Convert to string
    if isinstance(pattern, np.ndarray):
        binary_str = ''.join(map(str, pattern))
    else:
        binary_str = ''.join(map(str, pattern))
    
    # Extract segment number (first 4 bits) and copy index (last 4 bits)
    if len(binary_str) >= 8:
        segment_bits = binary_str[:4]
        copy_bits = binary_str[4:8]
        segment_number = int(segment_bits, 2)
        copy_index = int(copy_bits, 2)
        return segment_number, copy_index
    
    return None, None

def detect_patterns_in_segment(marked_file, expected_payload=None, segment_number=None, copy_index=None):
    """
    Detect patterns in each frame of a watermarked segment
    
    Args:
        marked_file: Path to watermarked segment
        expected_payload: Expected watermark payload
        segment_number: Segment number to derive expected payload if not provided
        copy_index: Copy index to use when deriving expected payload
    
    Returns:
        tuple: (most_common_pattern, frequency, success, detected_segment_number, detected_copy_index)
    """
    logger.info(f"Detecting patterns in {marked_file}")
    
    # If segment_number is provided but expected_payload isn't, derive it
    if expected_payload is None and segment_number is not None:
        expected_payload = generate_payload_for_segment(segment_number, copy_index or 0)
    
    # If neither is provided, try to extract segment number from filename
    if expected_payload is None and segment_number is None:
        segment_number = extract_segment_number_from_filename(marked_file)
        if segment_number is not None:
            expected_payload = generate_payload_for_segment(segment_number, copy_index or 0)
    
    # If we still don't have an expected payload, we can't verify
    if expected_payload is None:
        logger.warning(f"No expected payload for {marked_file}, cannot verify")
        return None, 0, False, None, None
    
    r = FileDecoder(marked_file)
    degenerator = DeShuffler(key=0)
    degenerator.set_shape(expected_payload.shape if isinstance(expected_payload, np.ndarray) else np.array(expected_payload).shape)
    frame_extractor = DwtDctSvdDecoder()
    
    # Use our custom pattern collector
    pattern_collector = PatternCollectorExtractor(r, frame_extractor, degenerator)
    most_common_pattern, frequency = pattern_collector.start()
    
    if most_common_pattern is None:
        logger.warning(f"No patterns found in {marked_file}")
        return None, 0, False, None, None
    
    # Check if most common pattern matches expected payload
    if isinstance(expected_payload, np.ndarray):
        success = np.array_equal(most_common_pattern, expected_payload)
    else:
        # If expected_payload is a list (from JSON), convert to numpy array
        success = np.array_equal(most_common_pattern, np.array(expected_payload))
    
    # Decode the detected pattern to segment number and copy index
    detected_segment_number, detected_copy_index = decode_watermark_pattern(most_common_pattern)
    
    logger.info(f"Segment {segment_number}: most common pattern: {most_common_pattern}, "
                f"expected: {expected_payload}, frequency: {frequency:.2f}, matches: {success}, "
                f"detected segment: {detected_segment_number}, copy: {detected_copy_index}")
    
    return most_common_pattern, frequency, success, detected_segment_number, detected_copy_index

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

def load_segment_copies(copies_file):
    """
    Load segment copies information from a JSON file
    
    Args:
        copies_file: Path to segment_copies.json file
        
    Returns:
        dict: Segment copies information
    """
    if not os.path.exists(copies_file):
        logger.warning(f"Copies file {copies_file} not found.")
        return {}
    
    with open(copies_file, 'r') as f:
        segment_copies = json.load(f)
    
    return segment_copies

def main():
    parser = argparse.ArgumentParser(description='Detect watermarks in leaked video')
    parser.add_argument('input_file', help='Path to input leaked video file')
    parser.add_argument('output_dir', help='Directory to store detection results')
    parser.add_argument('--payload-file', help='Path to segment_payloads.json from watermarking process')
    parser.add_argument('--copies-file', help='Path to segment_copies.json from watermarking process')
    parser.add_argument('--segment-duration', type=float, default=2.0, 
                        help='Duration of each segment in seconds (default: 2.0)')
    parser.add_argument('--clean', action='store_true', 
                        help='Clean output directory before processing')
    parser.add_argument('--max-copies', type=int, default=3,
                        help='Maximum number of copies to check for each segment (default: 3)')
    args = parser.parse_args()
    
    # Check if output_dir is inside the watermarking output directory
    # If the output_dir is 'detection' or starts with 'detection/',
    # and we have a path for copies_file, put detection inside that folder structure
    if args.copies_file and (args.output_dir == 'detection' or args.output_dir.startswith('detection/')):
        watermark_base_dir = os.path.dirname(os.path.abspath(args.copies_file))
        output_dir = os.path.join(watermark_base_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    
    segments_dir = os.path.join(output_dir, 'segments')
    
    # Clean directory if requested
    if args.clean and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create required directories
    for directory in [output_dir, segments_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Load segment payload mappings if available
    payload_file = args.payload_file
    segment_payloads = load_payload_mappings(payload_file) if payload_file else {}
    
    # Load segment copies information if available
    copies_file = args.copies_file
    segment_copies_info = load_segment_copies(copies_file) if copies_file else {}
    
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
        detected_copy = None
        best_match_frequency = 0
        
        # First, check using the payload mappings if available
        if segment_payloads:
            # Try to find the copy that matches best by checking all possible copies
            for copy_index in range(args.max_copies):
                # Try to find expected payload for this segment+copy
                expected_payload = segment_payloads.get(f"{segment_number}_{copy_index}")
                if expected_payload is None:
                    continue
                
                most_common_pattern, frequency, success, detected_segment, detected_copy_index = detect_patterns_in_segment(
                    segment, expected_payload, segment_number, copy_index)
                
                if success and frequency > best_match_frequency:
                    best_match_frequency = frequency
                    detected_copy = copy_index
        else:
            # If no payload mappings, just try to detect the pattern and decode it
            most_common_pattern, frequency, _, detected_segment, detected_copy_index = detect_patterns_in_segment(
                segment, None, segment_number)
            
            # If we were able to decode a segment number and copy index
            if detected_segment is not None and detected_copy_index is not None:
                # Check if detected segment matches the expected one
                if detected_segment == segment_number % 16:  # Match with modulo for segment numbers > 15
                    detected_copy = detected_copy_index
                    best_match_frequency = frequency
        
        # Record the results
        segment_results.append({
            'segment': segment,
            'segment_number': segment_number,
            'detected_copy_index': detected_copy,
            'match_frequency': best_match_frequency,
            'success': detected_copy is not None
        })
    
    # Step 4: Save and output results
    results_file = os.path.join(output_dir, 'detection_results.json')
    
    # Convert results to JSON-serializable format
    json_results = []
    for result in segment_results:
        json_results.append({
            'segment': os.path.basename(result['segment']),
            'segment_number': result['segment_number'],
            'detected_copy_index': result['detected_copy_index'],
            'match_frequency': result['match_frequency'],
            'success': result['success']
        })
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print detailed results for segments
    print("\n===== WATERMARK DETECTION RESULTS =====")
    for result in segment_results:
        segment_num = result['segment_number']
        copy_index = result['detected_copy_index']
        print(f"Segment {segment_num} ({os.path.basename(result['segment'])}):")
        if copy_index is not None:
            print(f"  Detected copy index: {copy_index}")
            print(f"  Match frequency: {result['match_frequency']:.2f}")
        else:
            print("  No watermark copy identified")
        print("-------------------------------")
    
    # Calculate overall success rate
    success_count = sum(1 for r in segment_results if r['success'])
    success_rate = success_count / len(segment_results) if segment_results else 0
    
    print("\n===== DETECTION SUMMARY =====")
    print(f"Total segments: {len(segment_results)}")
    print(f"Successfully identified copy indexes: {success_count}")
    print(f"Success rate: {success_rate*100:.2f}%")
    
    # Ordered sequence of copy indexes
    print("\n===== WATERMARKED COPY SEQUENCE =====")
    ordered_results = sorted(segment_results, key=lambda r: r['segment_number'] if r['segment_number'] is not None else float('inf'))
    
    copy_sequence = []
    for result in ordered_results:
        copy_index = result['detected_copy_index']
        if copy_index is not None:
            copy_sequence.append(copy_index)
            print(f"Segment {result['segment_number']}: Copy {copy_index}")
        else:
            copy_sequence.append(None)
            print(f"Segment {result['segment_number']}: Unknown copy")
    
    print("\n===== FINGERPRINT SEQUENCE =====")
    print(f"Copy sequence: {copy_sequence}")
    
    # Create a compact fingerprint from the copy sequence
    if all(copy is not None for copy in copy_sequence):
        fingerprint = ''.join([str(copy) for copy in copy_sequence])
        print(f"Copy fingerprint: {fingerprint}")
    
    logger.info(f"Detection results saved to {results_file}")
    
    return json_results

if __name__ == '__main__':
    main() 