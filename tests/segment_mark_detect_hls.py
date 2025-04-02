import logging
import os
import subprocess
import shutil
import numpy as np
import cv2
from pathlib import Path
from collections import Counter, defaultdict
import tempfile
import time

from offmark.embed.dwt_dct_svd_encoder import DwtDctSvdEncoder
from offmark.generator.shuffler import Shuffler
from offmark.video.embedder import Embedder
from offmark.video.frame_reader import FileDecoder
from offmark.video.frame_writer import FileEncoder
from offmark.degenerator.de_shuffler import DeShuffler
from offmark.extract.dwt_dct_svd_decoder import DwtDctSvdDecoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s')

# Directory structure
this_dir = os.path.dirname(__file__)
input_file = os.path.join(this_dir, 'media', 'in.mp4')

# Create directory structure for the whole workflow
base_dir = os.path.join(this_dir, 'watermark_test')
original_segments_dir = os.path.join(base_dir, 'original_segments')
marked_segments_dir = os.path.join(base_dir, 'marked_segments')
hls_dir = os.path.join(base_dir, 'hls')
reencoded_mp4_dir = os.path.join(base_dir, 'reencoded_mp4')
reencoded_segments_dir = os.path.join(base_dir, 'reencoded_segments')

# Create required directories
for directory in [base_dir, original_segments_dir, marked_segments_dir, 
                  hls_dir, reencoded_mp4_dir, reencoded_segments_dir]:
    os.makedirs(directory, exist_ok=True)

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
    
    return payload

# Modified custom extractor to collect patterns from each frame
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

def extract_segment_number_from_filename(filename):
    """
    Extract segment number from a filename
    
    Args:
        filename: The filename to parse
        
    Returns:
        int: The extracted segment number
    """
    # Extract the segment number from filenames like "segment_001.mp4" or "reencoded_segment_001.mp4"
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
    degenerator.set_shape(expected_payload.shape)
    frame_extractor = DwtDctSvdDecoder()
    
    # Use our custom pattern collector
    pattern_collector = PatternCollectorExtractor(r, frame_extractor, degenerator)
    most_common_pattern, frequency = pattern_collector.start()
    
    if most_common_pattern is None:
        logger.warning(f"No patterns found in {marked_file}")
        return None, 0, False
    
    # Check if most common pattern matches expected payload
    success = np.array_equal(most_common_pattern, expected_payload)
    
    logger.info(f"Segment {segment_number}: most common pattern: {most_common_pattern}, "
                f"expected: {expected_payload}, frequency: {frequency:.2f}, matches: {success}")
    
    return most_common_pattern, frequency, success

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

def convert_hls_to_mp4(playlist_file, output_mp4):
    """
    Convert HLS stream back to MP4
    
    Args:
        playlist_file: Path to HLS playlist file
        output_mp4: Path to output MP4 file
    """
    logger.info(f"Converting HLS stream {playlist_file} back to MP4 {output_mp4}")
    
    cmd = [
        'ffmpeg',
        '-i', playlist_file,
        '-c', 'copy',  # Use stream copy to avoid reencoding
        output_mp4
    ]
    
    subprocess.run(cmd, check=True)
    
    return output_mp4

def analyze_results(original_results, reencoded_results):
    """
    Compare the watermark detection results before and after HLS conversion
    
    Args:
        original_results: Results from original watermarked segments
        reencoded_results: Results from re-encoded segments
        
    Returns:
        dict: Analysis metrics
    """
    # Count successful detections
    original_success = sum(1 for r in original_results if r['success'])
    reencoded_success = sum(1 for r in reencoded_results if r['success'])
    
    # Calculate average pattern frequencies
    original_freq = np.mean([r['frequency'] for r in original_results])
    reencoded_freq = np.mean([r['frequency'] for r in reencoded_results])
    
    # Calculate success rates
    original_success_rate = original_success / len(original_results) if original_results else 0
    reencoded_success_rate = reencoded_success / len(reencoded_results) if reencoded_results else 0
    
    # Calculate watermark preservation rate through the HLS conversion process
    preservation_rate = reencoded_success / original_success if original_success else 0
    
    # Segment-by-segment matching (for segments that exist in both sets)
    segment_matches = 0
    segment_pairs = 0
    segment_preservation = defaultdict(dict)
    
    # Match segments by their number
    for original in original_results:
        original_num = extract_segment_number_from_filename(original['segment'])
        if original_num is not None:
            for reencoded in reencoded_results:
                reencoded_num = extract_segment_number_from_filename(reencoded['segment'])
                if reencoded_num is not None and original_num == reencoded_num:
                    segment_pairs += 1
                    if original['success'] and reencoded['success']:
                        segment_matches += 1
                    
                    segment_preservation[original_num] = {
                        'original_pattern': original['pattern'].tolist() if original['pattern'] is not None else None,
                        'original_success': original['success'],
                        'reencoded_pattern': reencoded['pattern'].tolist() if reencoded['pattern'] is not None else None,
                        'reencoded_success': reencoded['success'],
                        'preserved': original['success'] and reencoded['success']
                    }
    
    segment_preservation_rate = segment_matches / segment_pairs if segment_pairs else 0
    
    return {
        'original_success': original_success,
        'original_total': len(original_results),
        'original_success_rate': original_success_rate,
        'original_avg_frequency': original_freq,
        'reencoded_success': reencoded_success,
        'reencoded_total': len(reencoded_results),
        'reencoded_success_rate': reencoded_success_rate,
        'reencoded_avg_frequency': reencoded_freq,
        'preservation_rate': preservation_rate,
        'segment_matches': segment_matches,
        'segment_pairs': segment_pairs,
        'segment_preservation_rate': segment_preservation_rate,
        'segment_preservation': segment_preservation
    }

def run():
    """Main process to segment, mark, convert to HLS, and detect watermarks"""
    
    # Clean up previous runs
    shutil.rmtree(base_dir, ignore_errors=True)
    for directory in [base_dir, original_segments_dir, marked_segments_dir, 
                      hls_dir, reencoded_mp4_dir, reencoded_segments_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Segment the original video
    logger.info("\n===== STEP 1: SEGMENTING ORIGINAL VIDEO =====")
    segments = segment_video(input_file, os.path.join(original_segments_dir, 'segment_%03d.mp4'))
    logger.info(f"Created {len(segments)} segments")
    
    # Step 2: Watermark each segment with a unique pattern
    logger.info("\n===== STEP 2: WATERMARKING SEGMENTS WITH UNIQUE PATTERNS =====")
    marked_segments = []
    segment_payloads = {}  # Keep track of which payload was used for each segment
    
    for i, segment in enumerate(segments):
        output_file = os.path.join(marked_segments_dir, f"marked_{os.path.basename(segment)}")
        segment_number = extract_segment_number_from_filename(segment)
        payload = watermark_segment(segment, output_file, segment_number)
        marked_segments.append(output_file)
        segment_payloads[segment_number] = payload
    
    # Step 3: Detect patterns in each marked segment (original watermark verification)
    logger.info("\n===== STEP 3: DETECTING WATERMARKS IN ORIGINAL SEGMENTS =====")
    original_segment_results = []
    
    for marked_segment in marked_segments:
        segment_number = extract_segment_number_from_filename(marked_segment)
        expected_payload = segment_payloads.get(segment_number)
        most_common_pattern, frequency, success = detect_patterns_in_segment(marked_segment, expected_payload, segment_number)
        original_segment_results.append({
            'segment': os.path.basename(marked_segment),
            'segment_number': segment_number,
            'expected_payload': expected_payload,
            'pattern': most_common_pattern,
            'frequency': frequency,
            'success': success
        })
    
    # Step 4: Convert marked segments to HLS
    logger.info("\n===== STEP 4: CONVERTING MARKED SEGMENTS TO HLS =====")
    master_playlist, playlist = convert_segments_to_hls(marked_segments, hls_dir)
    
    # Step 5: Convert HLS back to MP4
    logger.info("\n===== STEP 5: CONVERTING HLS BACK TO MP4 =====")
    reencoded_mp4 = os.path.join(reencoded_mp4_dir, 'reencoded.mp4')
    convert_hls_to_mp4(playlist, reencoded_mp4)
    
    # Step 6: Re-segment the new MP4
    logger.info("\n===== STEP 6: RE-SEGMENTING THE REENCODED MP4 =====")
    reencoded_segments = segment_video(reencoded_mp4, os.path.join(reencoded_segments_dir, 'reencoded_segment_%03d.mp4'))
    
    # Step 7: Detect patterns in re-encoded segments
    logger.info("\n===== STEP 7: DETECTING WATERMARKS IN REENCODED SEGMENTS =====")
    reencoded_segment_results = []
    
    for reencoded_segment in reencoded_segments:
        segment_number = extract_segment_number_from_filename(reencoded_segment)
        expected_payload = segment_payloads.get(segment_number)
        most_common_pattern, frequency, success = detect_patterns_in_segment(reencoded_segment, expected_payload, segment_number)
        reencoded_segment_results.append({
            'segment': os.path.basename(reencoded_segment),
            'segment_number': segment_number,
            'expected_payload': expected_payload,
            'pattern': most_common_pattern,
            'frequency': frequency,
            'success': success
        })
    
    # Step 8: Analyze and report results
    logger.info("\n===== STEP 8: ANALYZING RESULTS =====")
    analysis = analyze_results(original_segment_results, reencoded_segment_results)
    
    # Print detailed results for original segments
    print("\n===== ORIGINAL SEGMENT PATTERN ANALYSIS =====")
    for result in original_segment_results:
        print(f"Segment {result['segment_number']} ({result['segment']}):")
        print(f"  Expected pattern : {result['expected_payload']}")
        print(f"  Detected pattern: {result['pattern']}")
        print(f"  Pattern frequency: {result['frequency']:.2f}")
        print(f"  Matches expected: {result['success']}")
        print("-------------------------------")
    
    # Print detailed results for re-encoded segments
    print("\n===== REENCODED SEGMENT PATTERN ANALYSIS =====")
    for result in reencoded_segment_results:
        print(f"Segment {result['segment_number']} ({result['segment']}):")
        print(f"  Expected pattern : {result['expected_payload']}")
        print(f"  Detected pattern: {result['pattern']}")
        print(f"  Pattern frequency: {result['frequency']:.2f}")
        print(f"  Matches expected: {result['success']}")
        print("-------------------------------")
    
    # Print overall comparison
    print("\n===== WATERMARK DURABILITY ANALYSIS =====")
    print(f"Original segments watermark detection rate: {analysis['original_success']}/{analysis['original_total']} ({analysis['original_success_rate']*100:.2f}%)")
    print(f"Original average pattern frequency: {analysis['original_avg_frequency']:.2f}")
    print(f"Reencoded segments watermark detection rate: {analysis['reencoded_success']}/{analysis['reencoded_total']} ({analysis['reencoded_success_rate']*100:.2f}%)")
    print(f"Reencoded average pattern frequency: {analysis['reencoded_avg_frequency']:.2f}")
    print(f"Watermark preservation rate through conversion: {analysis['preservation_rate']*100:.2f}%")
    print(f"Segment-by-segment preservation rate: {analysis['segment_preservation_rate']*100:.2f}%")
    
    # Print segment-by-segment preservation details
    print("\n===== SEGMENT-BY-SEGMENT PRESERVATION =====")
    for segment_num, data in sorted(analysis['segment_preservation'].items()):
        print(f"Segment {segment_num}: {'✅ Preserved' if data['preserved'] else '❌ Lost'}")
    
    # Overall success determination
    is_successful = analysis['segment_preservation_rate'] >= 0.75  # Consider success if 75% or more watermarks preserved
    
    return is_successful

if __name__ == '__main__':
    start_time = time.time()
    success = run()
    elapsed_time = time.time() - start_time
    
    print("\n===== FINAL CONCLUSION =====")
    if success:
        print(f"✅ Watermarking process completed successfully and survived HLS conversion!")
    else:
        print(f"❌ Watermarks did not survive HLS conversion at acceptable rates.")
    print(f"Total processing time: {elapsed_time:.2f} seconds") 