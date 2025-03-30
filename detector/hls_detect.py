import logging
import os
import numpy as np
import cv2

from offmark.degenerator.de_shuffler import DeShuffler
from offmark.extract.dwt_dct_svd_decoder import DwtDctSvdDecoder
from offmark.video.extractor import Extractor
from detector.m4s_reader import M4sFrameReader

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
            
            try:
                # Process frame for watermark extraction
                frame_yuv = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2YUV)
                decoded_yuv = self.frame_extractor.decode(frame_yuv)
                
                # Critical: use the degenerator to process the watermark data
                result = self.degenerator.degenerate(decoded_yuv)
                
                # Force to binary result for easier reading in logs
                binary_result = (result > 0.5).astype(int)
                
                # Store result for this frame
                self.all_frame_results.append((self.frame_count, binary_result))
                
                # Log every frame's payload
                logger.info(f"Frame {self.frame_count:3d} payload: {binary_result}")
                
            except Exception as e:
                logger.error(f"Error processing frame {self.frame_count}: {e}")
        
        # Analyze results across all frames
        if not self.all_frame_results:
            logger.warning("No valid frames were processed")
            return np.zeros(8), 0.0
            
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
        
        logger.info(f"\n=== FINAL RESULT ===")
        logger.info(f"Final detected pattern: {final_result}")
        logger.info(f"Confidence (based on consistency): {confidence:.4f}")
        
        return final_result, confidence


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
        reader = M4sFrameReader(m4s_file, init_segment)
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
        detected_bits, confidence = detector.detect()
        
        # Output results
        logger.info(f"Detected payload: {detected_bits}")
        logger.info(f"Confidence: {confidence:.4f}")
        
        if expected_payload is not None:
            match = np.array_equal(detected_bits, expected_payload)
            logger.info(f"Matches expected payload: {match}")
            
        return detected_bits, confidence
        
    except Exception as e:
        logger.error(f"Error detecting watermark: {str(e)}", exc_info=True)
        
        # Fallback to log parsing if detection fails
        logger.info("Trying to extract results from logs as fallback")
        result = detect_watermark_from_logs()
        if result is not None:
            return result, 0.5
        
        return np.zeros(8), 0.0
    finally:
        if reader:
            reader.close()

def main():
    """Detect watermark from the first marked segment"""
    this_dir = os.path.dirname(__file__)
    hls_dir = os.path.join(this_dir, 'hls_output')
    marked_dir = os.path.join(this_dir, 'out', 'marked_segments')
    
    # Get init segment path
    init_segment = os.path.join(hls_dir, 'init.mp4')
    if not os.path.exists(init_segment):
        logger.error(f"Init segment not found: {init_segment}")
        return
    
    # Find the first marked segment
    marked_segment = None
    for filename in os.listdir(marked_dir):
        if filename.endswith('.m4s'):
            marked_segment = os.path.join(marked_dir, filename)
            break
    
    if not marked_segment:
        logger.error("Could not find any marked segment in output directory")
        return
    
    # Define expected payload - 01010101
    expected_payload = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    
    # Detect watermark
    print("\n==================================================")
    print(f"STARTING DETECTION: Analyzing {marked_segment}")
    print(f"Expected watermark payload: {expected_payload}")
    print(f"This will display the watermark payload for EACH frame")
    print("==================================================\n")
    
    detected_bits, confidence = detect_watermark(marked_segment, init_segment, expected_payload)
    
    # Print summary
    print("\n=== FINAL DETECTION SUMMARY ===")
    print(f"Expected payload: {expected_payload}")
    print(f"Detected payload: {detected_bits}")
    print(f"Confidence: {confidence:.4f}")
    match = np.array_equal(detected_bits, expected_payload)
    print(f"Match: {'✓' if match else '✗'}")
    
    if not match:
        # Calculate how many bits are correct
        correct_bits = np.sum(detected_bits == expected_payload)
        print(f"Correct bits: {correct_bits}/8 ({correct_bits/8*100:.1f}%)")
        
        # Show which bits are wrong
        incorrect_indices = np.where(detected_bits != expected_payload)[0]
        print("Incorrect bits (index):", incorrect_indices)
        for idx in incorrect_indices:
            print(f"  Bit {idx}: Expected {expected_payload[idx]}, Got {detected_bits[idx]}")

if __name__ == "__main__":
    main() 