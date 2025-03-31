import logging
import os
import time
import numpy as np
import cv2
import argparse
import re

from offmark.embed.dwt_dct_svd_encoder import DwtDctSvdEncoder
from offmark.generator.shuffler import Shuffler
from offmark.video.embedder import Embedder
from m4s_frame_reader import M4sFrameReader
from m4s_frame_writer import M4sFrameWriter, DebugM4sFrameWriter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s')


def extract_segment_number(filename):
    """Extract segment number from m4s filename."""
    match = re.search(r'segment[_-](\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def parse_m3u8(playlist_path):
    """Parse the M3U8 playlist to get segment information."""
    segments = []
    with open(playlist_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.endswith('.m4s'):
            segments.append(line)
    
    return segments


def int_to_binary_array(number, bits=8):
    """Convert an integer to binary numpy array."""
    binary = format(number, f'0{bits}b')
    return np.array([int(b) for b in binary])


def mark_hls_segment(segment_file, init_segment, output_file, payload, enable_debug=False, debug_dir=None):
    """Watermark a single HLS segment with the given payload"""
    start_time = time.time()
    
    logger.info(f"Using payload: {payload}")
    
    # Create reader and writer
    reader = None
    writer = None
    
    try:
        # Initialize reader with M4S segment
        logger.info(f"Creating reader for segment: {segment_file}")
        reader = M4sFrameReader(segment_file, init_segment)
        logger.info(f"Reader created, dimensions: {reader.width}x{reader.height}")
        
        # Initialize writer with appropriate M4S writer
        logger.info(f"Creating M4S writer for output file: {output_file}")
        if enable_debug and debug_dir:
            writer = DebugM4sFrameWriter(
                output_file, 
                reader.width, 
                reader.height,
                debug_dir,
                prefix=f"wm_seg_{extract_segment_number(segment_file) or 0}",
                save_debug_frames=True
            )
            logger.info("Debug M4S writer created (with debug frame saving)")
        else:
            # Use basic M4sFrameWriter with pipe-based approach
            writer = M4sFrameWriter(output_file, reader.width, reader.height)
            logger.info("M4S writer created (no debug frame saving)")
        
        # Check if writer had initialization errors or is not alive
        if not writer.is_alive():
            error = writer.get_last_error() or "FFmpeg process terminated unexpectedly"
            logger.error(f"FFmpeg error during processing: {error}")
            return False
        
        # Initialize Frame Embedder with precise settings
        frame_embedder = DwtDctSvdEncoder()
        capacity = frame_embedder.wm_capacity((reader.height, reader.width, 3))
        logger.info(f"Watermark capacity: {capacity} bits")
        
        # Initialize Generator and create watermark
        generator = Shuffler(key=0)
        wm = generator.generate_wm(payload, capacity)
        frame_embedder.read_wm(wm)
        logger.info("Watermark prepared for embedding")
        
        # Manual frame processing for maximum quality control
        logger.info("Starting manual frame processing...")
        frame_count = 0
        
        while True:
            # Read frame
            in_frame = reader.read()
            if in_frame is None:
                logger.info(f"End of stream reached after {frame_count} frames")
                break
                
            # Process frame with careful handling of data types
            frame_count += 1
            
            # Convert to float32 for processing
            in_frame = in_frame.astype(np.float32)
            
            # Convert to YUV for embedding
            in_frame_yuv = cv2.cvtColor(in_frame, cv2.COLOR_BGR2YUV)
            
            # Embed watermark
            out_frame_yuv = frame_embedder.encode(in_frame_yuv)
            
            # Convert back to BGR
            out_frame = cv2.cvtColor(out_frame_yuv, cv2.COLOR_YUV2BGR)
            
            # Clip to valid range and convert to uint8 for output
            out_frame = np.clip(out_frame, 0.0, 255.0)
            out_frame = np.round(out_frame).astype(np.uint8)
            
            # Write frame
            writer.write(out_frame)
            
            # Log progress
            if frame_count % 10 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        logger.info(f"Completed processing {frame_count} frames")
        
        # Check for any writer errors after completion
        if not writer.is_alive():
            error = writer.get_last_error() or "FFmpeg process terminated unexpectedly"
            logger.error(f"FFmpeg error during processing: {error}")
            return False
            
        # Check if output file was created successfully
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logger.error(f"Output file is missing or empty: {output_file}")
            return False
        
        end_time = time.time()
        logger.info(f"Watermarking completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Output file saved to: {output_file} ({os.path.getsize(output_file)/1024:.1f} KB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error watermarking segment: {str(e)}", exc_info=True)
        # Check if the writer had an error
        if writer and hasattr(writer, 'get_last_error'):
            error = writer.get_last_error()
            if error:
                logger.error(f"Writer error details: {error}")
        return False
    finally:
        # Clean up resources
        if reader:
            logger.debug("Closing reader")
            try:
                reader.close()
            except Exception as e:
                logger.error(f"Error closing reader: {str(e)}")
            
        if writer:
            logger.debug("Closing writer")
            try:
                writer.close()
            except Exception as e:
                logger.error(f"Error closing writer: {str(e)}")
            
            # Final check for writer errors after close
            if hasattr(writer, 'get_last_error'):
                error = writer.get_last_error()
                if error:
                    logger.error(f"Writer error after close: {error}")


def mark_all_hls_segments():
    """Watermark all segments in an HLS stream"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Watermark HLS stream segments')
    parser.add_argument('--start', type=int, help='Start segment number (inclusive)', default=0)
    parser.add_argument('--end', type=int, help='End segment number (inclusive)', default=999)
    parser.add_argument('--segments', type=str, help='Comma-separated list of specific segment numbers to process')
    parser.add_argument('--skip-existing', action='store_true', help='Skip segments that already exist in output')
    parser.add_argument('--payload', type=str, help='Custom payload (8 bits, e.g. 01010101) for all segments', required=True)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (saves debug frames)')
    args = parser.parse_args()
    
    # Setup paths
    this_dir = os.path.dirname(__file__)
    hls_dir = os.path.join(this_dir, '..', 'tests', 'hls_output')
    output_dir = os.path.join(this_dir, '..', 'out', 'marked_segments')
    debug_dir = os.path.join(this_dir, '..', 'debug_watermarking') if args.debug else None
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if args.debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
    # Get init segment path
    init_segment = os.path.join(hls_dir, 'init.mp4')
    if not os.path.exists(init_segment):
        logger.error(f"Init segment not found: {init_segment}")
        return
    
    # Get playlist path
    playlist_path = os.path.join(hls_dir, 'playlist.m3u8')
    if not os.path.exists(playlist_path):
        logger.error(f"Playlist not found: {playlist_path}")
        return
    
    # Parse custom payload
    if len(args.payload) != 8 or not all(bit in '01' for bit in args.payload):
        logger.error("Custom payload must be exactly 8 bits (0s and 1s)")
        return
    custom_payload = np.array([int(bit) for bit in args.payload])
    logger.info(f"Using custom payload for all segments: {custom_payload}")
    
    # Get segments from playlist
    playlist_segments = parse_m3u8(playlist_path)
    logger.info(f"Found {len(playlist_segments)} segments in playlist")
    
    # Get list of segments to process
    segments_to_process = []
    
    for segment_filename in playlist_segments:
        segment_file = os.path.join(hls_dir, segment_filename)
        if not os.path.exists(segment_file):
            logger.warning(f"Segment file not found: {segment_file}")
            continue
            
        segment_number = extract_segment_number(segment_file)
        if segment_number is None:
            logger.warning(f"Could not extract segment number from {segment_file}, skipping...")
            continue
        
        # Filter segments based on command line arguments
        if args.segments:
            segment_list = [int(s.strip()) for s in args.segments.split(',')]
            if segment_number not in segment_list:
                continue
        elif not (args.start <= segment_number <= args.end):
            continue
            
        # Check if output file already exists
        output_file = os.path.join(output_dir, os.path.basename(segment_file))
        if args.skip_existing and os.path.exists(output_file):
            logger.info(f"Skipping existing segment {segment_number}: {output_file}")
            continue
            
        segments_to_process.append((segment_number, segment_file, output_file))
    
    # Sort segments by number for sequential processing
    segments_to_process.sort()
    
    # Process each segment
    total_segments = len(segments_to_process)
    successful_segments = 0
    failed_segments = 0
    
    logger.info(f"=== STARTING WATERMARKING OF {total_segments} SEGMENTS ===")
    
    start_time = time.time()
    
    for i, (segment_number, segment_file, output_file) in enumerate(segments_to_process):
        progress = f"[{i+1}/{total_segments}]"
        logger.info(f"{progress} Processing segment {segment_number}: {segment_file}")
        
        # Process the segment with the custom payload
        success = mark_hls_segment(
            segment_file, 
            init_segment, 
            output_file, 
            payload=custom_payload, 
            enable_debug=args.debug, 
            debug_dir=debug_dir
        )
        
        # Update counters
        if success:
            successful_segments += 1
            logger.info(f"{progress} Successfully watermarked segment {segment_number}")
        else:
            failed_segments += 1
            logger.error(f"{progress} Failed to watermark segment {segment_number}")
        
        # Print progress summary
        elapsed_time = time.time() - start_time
        avg_time_per_segment = elapsed_time / (i + 1)
        remaining_segments = total_segments - (i + 1)
        estimated_remaining_time = avg_time_per_segment * remaining_segments
        
        logger.info(f"Progress: {i+1}/{total_segments} segments processed")
        logger.info(f"Successful: {successful_segments}, Failed: {failed_segments}")
        logger.info(f"Time elapsed: {elapsed_time:.2f}s, Estimated remaining: {estimated_remaining_time:.2f}s")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n=== WATERMARKING COMPLETE ===")
    logger.info(f"Total segments: {total_segments}")
    logger.info(f"Successfully watermarked: {successful_segments}")
    logger.info(f"Failed to watermark: {failed_segments}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    return {
        'total_segments': total_segments,
        'successful_segments': successful_segments,
        'failed_segments': failed_segments,
        'total_time': total_time
    }


if __name__ == "__main__":
    result = mark_all_hls_segments()
    if result:
        print(f"\nWatermarking complete! Processed {result['successful_segments']} of {result['total_segments']} segments.") 