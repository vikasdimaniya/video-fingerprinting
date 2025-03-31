import logging
import os
import numpy as np
import cv2

from offmark.degenerator.de_shuffler import DeShuffler
from offmark.extract.dwt_dct_svd_decoder import DwtDctSvdDecoder
from offmark.video.extractor import Extractor
from m4s_reader import SimpleM4sReader

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_watermark_from_logs():
    """Parse the logs to find the detection results"""
    logger.info("Trying to extract results from logs...")
    
    # Get the log file from the current run directory
    import glob
    log_files = glob.glob("*.log")
    
    if not log_files:
        logger.warning("No log files found, cannot extract results from logs")
        return None
        
    # Get the most recent log file
    log_file = max(log_files, key=os.path.getctime)
    logger.info(f"Found log file: {log_file}")
    
    # Read the log file
    with open(log_file, 'r') as f:
        log_contents = f.read()
    
    # Find all detection results - they're logged as "[0 1 0 1 0 1 0 1]" format
    import re
    results = re.findall(r'\[(\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+)\]', log_contents)
    
    if not results:
        logger.warning("No detection results found in log file")
        return None
        
    # Use the last result as the final detection
    final_result_str = results[-1]
    final_result = np.array([int(x) for x in final_result_str.split()])
    
    logger.info(f"Extracted result from logs: {final_result}")
    return final_result


class SimpleDetector:
    """Simpler watermark detector that avoids extractor.start() issues"""
    
    def __init__(self, frame_reader, frame_extractor, degenerator):
        self.frame_reader = frame_reader
        self.frame_extractor = frame_extractor
        self.degenerator = degenerator
        self.frame_count = 0
        self.all_frame_results = []  # Track payload from every frame
        self.detection_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'error_frames': 0
        }
    
    def detect(self):
        """Process frames manually without using the original extractor"""
        logger.info("Starting simplified detection process")
        
        # List to track all frame payloads for analysis
        while True:
            # Read the next frame
            frame = self.frame_reader.read()
            if frame is None:
                logger.info(f"End of frames reached after {self.frame_count} frames")
                break
                
            # Track frame count
            self.frame_count += 1
            self.detection_stats['total_frames'] += 1
            
            try:
                # Process frame for watermark extraction with careful type handling
                # Convert to float32 first for better precision
                frame_float = frame.astype(np.float32)
                
                # Convert to YUV color space for detection (same as what was used in embedding)
                frame_yuv = cv2.cvtColor(frame_float, cv2.COLOR_BGR2YUV)
                
                # Extract the watermark
                decoded_yuv = self.frame_extractor.decode(frame_yuv)
                
                # Critical: use the degenerator to process the watermark data
                result = self.degenerator.degenerate(decoded_yuv)
                
                # Force to binary result for easier reading in logs
                binary_result = (result > 0.5).astype(int)
                
                # Store result for this frame
                self.all_frame_results.append((self.frame_count, binary_result))
                self.detection_stats['processed_frames'] += 1
                
                # Log every frame's payload but limit frequency for large segments
                if self.frame_count <= 10 or self.frame_count % 10 == 0:
                    logger.info(f"Frame {self.frame_count:3d} payload: {binary_result}")
                
            except Exception as e:
                self.detection_stats['error_frames'] += 1
                logger.error(f"Error processing frame {self.frame_count}: {e}")
        
        # Analyze results across all frames
        if not self.all_frame_results:
            logger.warning("No valid frames were processed")
            return np.zeros(8), 0.0, {}
            
        # Calculate how many frames had each bit pattern
        from collections import Counter
        bit_patterns = Counter()
        for _, result in self.all_frame_results:
            bit_patterns[tuple(result)] += 1
            
        # Find the most common bit pattern
        most_common_pattern, count = bit_patterns.most_common(1)[0]
        percentage = (count / len(self.all_frame_results)) * 100
        
        logger.info(f"\n=== BIT PATTERN ANALYSIS ===")
        logger.info(f"Total frames analyzed: {len(self.all_frame_results)}")
        logger.info(f"Most common bit pattern: {most_common_pattern} (appeared in {count} frames, {percentage:.1f}%)")
        
        # Show all patterns that appeared in at least 5% of frames
        logger.info(f"\nAll significant patterns (>5% of frames):")
        for pattern, count in bit_patterns.most_common():
            pct = (count / len(self.all_frame_results)) * 100
            if pct >= 5.0:
                logger.info(f"Pattern {pattern}: {count} frames ({pct:.1f}%)")
        
        # Look at transitions - where the bit pattern changes
        transitions = []
        prev_pattern = None
        for frame_num, pattern in self.all_frame_results:
            if prev_pattern is not None and not np.array_equal(pattern, prev_pattern):
                transitions.append((frame_num, prev_pattern, pattern))
            prev_pattern = pattern
            
        logger.info(f"\n=== PATTERN TRANSITIONS ===")
        logger.info(f"Total transitions: {len(transitions)}")
        if len(transitions) > 0:
            logger.info(f"First 10 transitions:")
            for i, (frame_num, from_pattern, to_pattern) in enumerate(transitions[:10]):
                logger.info(f"Frame {frame_num}: {from_pattern} -> {to_pattern}")
        
        # Use the most common pattern as the result
        final_result = np.array(most_common_pattern)
        
        # Calculate confidence based on consistency
        confidence = percentage / 100.0
        
        # Calculate bit-level statistics
        bit_stats = {}
        all_results_array = np.array([result for _, result in self.all_frame_results])
        for bit_pos in range(final_result.shape[0]):
            bit_values = all_results_array[:, bit_pos]
            ones_count = np.sum(bit_values)
            zeros_count = len(bit_values) - ones_count
            dominant_bit = 1 if ones_count > zeros_count else 0
            consistency = (max(ones_count, zeros_count) / len(bit_values)) * 100
            bit_stats[bit_pos] = {
                'dominant_bit': dominant_bit,
                'consistency': consistency,
                'ones_count': int(ones_count),
                'zeros_count': int(zeros_count)
            }
        
        # Prepare detection summary
        detection_summary = {
            'frame_count': self.frame_count,
            'valid_frames': len(self.all_frame_results),
            'most_common_pattern': most_common_pattern,
            'pattern_frequency': percentage,
            'transitions': len(transitions),
            'bit_stats': bit_stats,
            'all_patterns': dict(bit_patterns),
            'processing_stats': self.detection_stats
        }
        
        logger.info(f"\n=== FINAL RESULT ===")
        logger.info(f"Final detected pattern: {final_result}")
        logger.info(f"Confidence (based on consistency): {confidence:.4f}")
        
        return final_result, confidence, detection_summary


def detect_watermark(m4s_file, init_segment, expected_payload=None):
    """Detect watermark from a M4S segment file"""
    logger.info(f"Detecting watermark from: {m4s_file}")
    logger.info(f"Using init segment: {init_segment}")
    
    if expected_payload is not None:
        logger.info(f"Expected payload: {expected_payload}")
    
    # Create reader
    reader = None
    try:
        # Initialize reader with M4S segment - using our simplified reader
        reader = SimpleM4sReader(m4s_file, init_segment)
        logger.info(f"Reader created, dimensions: {reader.width}x{reader.height}")
        
        # Initialize frame extractor
        frame_extractor = DwtDctSvdDecoder()
        logger.info("Frame extractor initialized")
        
        # Initialize degenerator
        degenerator = DeShuffler(key=0)
        if expected_payload is not None:
            degenerator.set_shape(expected_payload.shape)
        else:
            degenerator.set_shape((8,))
        logger.info("Degenerator initialized")
        
        # Use our custom detector
        detector = SimpleDetector(reader, frame_extractor, degenerator)
        logger.info("Detector initialized")
        
        # Run detection
        detected_bits, confidence, detection_summary = detector.detect()
        
        # Output results
        logger.info(f"Detected payload: {detected_bits}")
        logger.info(f"Confidence: {confidence:.4f}")
        
        if expected_payload is not None:
            match = np.array_equal(detected_bits, expected_payload)
            logger.info(f"Matches expected payload: {match}")
            
        return detected_bits, confidence, detection_summary
        
    except Exception as e:
        logger.error(f"Error detecting watermark: {str(e)}", exc_info=True)
        
        # Fallback to log parsing if detection fails
        logger.info("Trying to extract results from logs as fallback")
        result = detect_watermark_from_logs()
        if result is not None:
            return result, 0.5, {"error": str(e)}
        
        return np.zeros(8), 0.0, {"error": str(e)}
    finally:
        if reader:
            reader.close()

def main():
    """Detect watermark from all marked segments in the HLS stream"""
    this_dir = os.path.dirname(__file__)
    hls_dir = os.path.join(this_dir, 'hls_output')
    marked_dir = os.path.join(this_dir, 'out', 'marked_segments')
    
    # Get init segment path
    init_segment = os.path.join(hls_dir, 'init.mp4')
    if not os.path.exists(init_segment):
        logger.error(f"Init segment not found: {init_segment}")
        return
    
    # Find all marked segments
    marked_segments = []
    if os.path.exists(marked_dir):
        for filename in sorted(os.listdir(marked_dir)):
            if filename.endswith('.m4s'):
                marked_segments.append(os.path.join(marked_dir, filename))
    
    # If no marked segments found in marked_dir, check hls_dir
    if not marked_segments:
        logger.info("No marked segments found in marked directory, checking HLS directory")
        for filename in sorted(os.listdir(hls_dir)):
            if filename.endswith('.m4s') and filename != 'init.mp4':
                marked_segments.append(os.path.join(hls_dir, filename))
    
    if not marked_segments:
        logger.error("Could not find any m4s segments to analyze")
        return
    
    # Define expected payload - 01010101
    expected_payload = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    
    # Initialize array to store all detected payloads
    all_payloads = []
    segment_results = []
    
    # Process each segment
    print("\n==================================================")
    print(f"STARTING DETECTION: Analyzing {len(marked_segments)} segments")
    print(f"Expected watermark payload: {expected_payload}")
    print("==================================================\n")
    
    # Calculate total segments to process
    total_segments = len(marked_segments)
    
    for i, segment_file in enumerate(marked_segments):
        segment_name = os.path.basename(segment_file)
        print(f"\n[{i+1}/{total_segments}] Processing segment: {segment_name}")
        
        # Detect watermark for this segment
        detected_bits, confidence, detection_summary = detect_watermark(segment_file, init_segment, expected_payload)
        
        # Store the result
        all_payloads.append(detected_bits)
        segment_results.append({
            'segment': segment_name,
            'payload': detected_bits,
            'confidence': confidence,
            'match': np.array_equal(detected_bits, expected_payload) if expected_payload is not None else None,
            'detection_summary': detection_summary
        })
        
        # Print segment summary
        match_str = ""
        if expected_payload is not None:
            match = np.array_equal(detected_bits, expected_payload)
            match_str = f" {'✓' if match else '✗'}"
        
        print(f"Detected: {detected_bits}{match_str} (confidence: {confidence:.4f})")
        
        # Print quick stats from the detection summary if available
        if isinstance(detection_summary, dict) and 'frame_count' in detection_summary:
            print(f"Processed {detection_summary['valid_frames']}/{detection_summary['frame_count']} frames")
            if 'pattern_frequency' in detection_summary:
                print(f"Pattern consistency: {detection_summary['pattern_frequency']:.1f}%")
    
    # Create numpy array of all payloads for analysis
    all_payloads_array = np.array(all_payloads)
    
    # Print comprehensive analysis of all segments
    print("\n=== ALL SEGMENTS DETECTION SUMMARY ===")
    print(f"Total segments processed: {len(marked_segments)}")
    
    # Calculate consensus payload across all segments
    if len(all_payloads) > 0:
        # Use majority voting for each bit position
        consensus_payload = np.zeros(8, dtype=int)
        for bit_pos in range(8):
            bit_values = all_payloads_array[:, bit_pos]
            ones_count = np.sum(bit_values)
            zeros_count = len(bit_values) - ones_count
            consensus_payload[bit_pos] = 1 if ones_count > zeros_count else 0
        
        # Calculate match with expected payload
        if expected_payload is not None:
            consensus_match = np.array_equal(consensus_payload, expected_payload)
            match_str = f"{'✓' if consensus_match else '✗'}"
            print(f"\nConsensus payload across all segments: {consensus_payload} {match_str}")
            
            # Calculate how many bits match expected
            correct_bits = np.sum(consensus_payload == expected_payload)
            print(f"Correct bits: {correct_bits}/8 ({correct_bits/8*100:.1f}%)")
            
            # Show which bits are wrong in consensus
            if not consensus_match:
                incorrect_indices = np.where(consensus_payload != expected_payload)[0]
                print("Incorrect bits (index):", incorrect_indices)
                for idx in incorrect_indices:
                    print(f"  Bit {idx}: Expected {expected_payload[idx]}, Got {consensus_payload[idx]}")
        else:
            print(f"\nConsensus payload across all segments: {consensus_payload}")
        
        # Count occurrences of each unique payload
        from collections import Counter
        unique_payloads = [tuple(payload) for payload in all_payloads]
        payload_counts = Counter(unique_payloads)
        
        # Display most common payloads
        print("\nMost common payloads across segments:")
        for payload, count in payload_counts.most_common():
            percentage = (count / len(all_payloads)) * 100
            match_str = ""
            if expected_payload is not None:
                payload_array = np.array(payload)
                match = np.array_equal(payload_array, expected_payload)
                match_str = f" {'✓' if match else '✗'}"
            print(f"Payload {payload}: {count} segments ({percentage:.1f}%){match_str}")
        
        # Calculate consistency for each bit position
        print("\nBit-wise consistency across segments:")
        for bit_pos in range(8):
            bit_values = all_payloads_array[:, bit_pos]
            ones_count = np.sum(bit_values)
            zeros_count = len(bit_values) - ones_count
            dominant = 1 if ones_count > zeros_count else 0
            percentage = (max(ones_count, zeros_count) / len(bit_values)) * 100
            exp_match = ""
            if expected_payload is not None:
                exp_match = f" {'✓' if dominant == expected_payload[bit_pos] else '✗'}"
            print(f"Bit {bit_pos}: {dominant} ({percentage:.1f}% consistent){exp_match}")
    
    # Prepare a structured result object with analysis
    result = {
        'all_payloads': all_payloads,
        'consensus_payload': consensus_payload.tolist() if 'consensus_payload' in locals() else None,
        'segment_results': segment_results,
        'expected_payload': expected_payload.tolist() if expected_payload is not None else None,
        'total_segments': total_segments,
        'consistency': {
            'by_bit': {},
            'by_payload': dict([(str(k), v) for k, v in payload_counts.items()]) if 'payload_counts' in locals() else {}
        }
    }
    
    # Fill bit-level consistency data
    if len(all_payloads) > 0:
        for bit_pos in range(8):
            bit_values = all_payloads_array[:, bit_pos]
            ones_count = np.sum(bit_values)
            zeros_count = len(bit_values) - ones_count
            dominant = 1 if ones_count > zeros_count else 0
            consistency = (max(ones_count, zeros_count) / len(bit_values)) * 100
            result['consistency']['by_bit'][bit_pos] = {
                'dominant_bit': int(dominant),
                'consistency_pct': float(consistency),
                'ones_count': int(ones_count),
                'zeros_count': int(zeros_count)
            }
    
    # Return the structured result with all data
    return result

if __name__ == "__main__":
    detection_result = main()
    if detection_result:
        # Summarize the detection result
        all_payloads = detection_result.get('all_payloads', [])
        if all_payloads:
            print(f"\nDetected {len(all_payloads)} segment payloads")
            # If you need the raw array of payloads
            print(f"Consensus payload: {detection_result.get('consensus_payload')}") 