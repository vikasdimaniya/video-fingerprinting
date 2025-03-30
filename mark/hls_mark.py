import logging
import os
import time
import numpy as np
import cv2

from offmark.embed.dwt_dct_svd_encoder import DwtDctSvdEncoder
from offmark.generator.shuffler import Shuffler
from offmark.video.embedder import Embedder
from m4s_frame_reader import M4sFrameReader
from m4s_frame_writer import DebugM4sFrameWriter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s')


def mark_hls_segment():
    """Watermark the first segment of HLS stream with 01010101 payload"""
    start_time = time.time()
    
    # Setup paths
    this_dir = os.path.dirname(__file__)
    hls_dir = os.path.join(this_dir, 'hls_output')
    output_dir = os.path.join(this_dir, 'out', 'marked_segments')
    debug_dir = os.path.join(this_dir, 'debug_watermarking')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Get init segment path
    init_segment = os.path.join(hls_dir, 'init.mp4')
    if not os.path.exists(init_segment):
        logger.error(f"Init segment not found: {init_segment}")
        return
    
    # Find segment 0
    segment_file = None
    for filename in os.listdir(hls_dir):
        if filename.endswith('.m4s') and ('segment_000' in filename or 'segment-0' in filename):
            segment_file = os.path.join(hls_dir, filename)
            break
    
    if not segment_file:
        logger.error("Could not find segment 0 in HLS directory")
        return
    
    # Set output file
    output_file = os.path.join(output_dir, os.path.basename(segment_file))
    
    # Define payload - 01010101
    payload = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    logger.info(f"Payload: {payload}")
    
    # Create reader and writer
    reader = None
    writer = None
    
    try:
        # Initialize reader with M4S segment
        logger.info(f"Creating reader for segment: {segment_file}")
        reader = M4sFrameReader(segment_file, init_segment)
        logger.info(f"Reader created, dimensions: {reader.width}x{reader.height}")
        
        # Initialize writer with our custom M4S writer
        logger.info(f"Creating M4S writer for output file: {output_file}")
        writer = DebugM4sFrameWriter(
            output_file, 
            reader.width, 
            reader.height,
            debug_dir,
            prefix="watermarked"
        )
        logger.info("Debug M4S writer created")
        
        # Initialize Frame Embedder
        frame_embedder = DwtDctSvdEncoder()
        capacity = frame_embedder.wm_capacity((reader.height, reader.width, 3))
        logger.info(f"Watermark capacity: {capacity} bits")
        
        # Initialize Generator and create watermark
        generator = Shuffler(key=0)
        wm = generator.generate_wm(payload, capacity)
        frame_embedder.read_wm(wm)
        logger.info("Watermark prepared for embedding")
        
        # Create embedder and start the process
        logger.info("Starting watermarking process...")
        video_embedder = Embedder(reader, frame_embedder, writer)
        video_embedder.start()
        
        end_time = time.time()
        logger.info(f"Watermarking completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Output file saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error watermarking segment: {str(e)}", exc_info=True)
    finally:
        # Clean up resources
        if reader:
            logger.debug("Closing reader")
            reader.close()
            
        if writer:
            logger.debug("Closing writer")
            writer.close()


if __name__ == "__main__":
    mark_hls_segment() 