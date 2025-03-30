import cv2
import os
import numpy as np
import logging
import tempfile
import shutil
import subprocess
from offmark.video.frame_reader import FrameReader

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class M4sFrameReader(FrameReader):
    """M4S reader that reads frames directly from FFmpeg pipe"""
    
    def __init__(self, m4s_file, init_segment):
        super().__init__()
        self.m4s_file = m4s_file
        self.init_segment = init_segment
        self.temp_dir = None
        self.width = None
        self.height = None
        self.ffmpeg_process = None
        self.frame_count = 0
        
        # Temporary files are only needed for the concat file, not for the frames
        self.temp_dir = tempfile.mkdtemp(prefix="m4s_pipe_")
        
        # Start the FFmpeg process in the constructor
        self._start_ffmpeg()
    
    def _start_ffmpeg(self):
        """Start FFmpeg process with pipe for direct frame reading"""
        logger.info(f"Setting up FFmpeg pipe for {self.m4s_file}")
        
        try:
            # Step 1: Create concat file for FFmpeg
            concat_file = os.path.join(self.temp_dir, "concat.txt")
            with open(concat_file, 'w') as f:
                f.write(f"file '{os.path.abspath(self.init_segment)}'\n")
                f.write(f"file '{os.path.abspath(self.m4s_file)}'\n")
            
            logger.debug(f"Created concat file at {concat_file}")
            
            # Step 2: Start FFmpeg process with raw video output piped to stdout
            # Using rawvideo format for direct access to pixel data
            ffmpeg_cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-vsync", "0",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
                "-pix_fmt", "bgr24",      # Use BGR format like OpenCV expects
                "-f", "rawvideo",         # Output raw video frames
                "-"                       # Output to stdout
            ]
            
            logger.debug(f"Starting FFmpeg pipe: {' '.join(ffmpeg_cmd)}")
            
            # Start FFmpeg with pipe for stdout
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Use a large buffer for better performance
            )
            
            # Read first frame to get dimensions
            first_frame = self.read()
            if first_frame is not None:
                self.height, self.width = first_frame.shape[:2]
                logger.info(f"Frame dimensions: {self.width}x{self.height}")
                # Put the frame back (we'll need to implement this)
                self._rewind_one_frame(first_frame)
            else:
                logger.warning("Could not read first frame, using default dimensions")
                self.width = 1280
                self.height = 720
                
        except Exception as e:
            logger.error(f"Error starting FFmpeg process: {str(e)}", exc_info=True)
            self.cleanup()
            raise
    
    def _rewind_one_frame(self, frame):
        """Save the first frame so we don't lose it during dimension detection"""
        self.cached_first_frame = frame
        self.frame_count -= 1  # Compensate for the frame we just read
    
    def read(self):
        """Read the next frame directly from FFmpeg pipe"""
        # If we have a cached frame from dimension detection, return it
        if hasattr(self, 'cached_first_frame') and self.frame_count == 0:
            frame = self.cached_first_frame
            delattr(self, 'cached_first_frame')
            self.frame_count += 1
            return frame
            
        # Check if FFmpeg process is still running
        if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
            if self.ffmpeg_process and self.ffmpeg_process.poll() != 0:
                logger.error(f"FFmpeg process exited with code: {self.ffmpeg_process.poll()}")
                stderr = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg error: {stderr}")
            logger.debug("FFmpeg process not running, end of frames")
            return None
        
        try:
            # Calculate frame size in bytes (height * width * 3 channels for BGR)
            if self.width is None or self.height is None:
                # If we don't know dimensions yet, use default and try to read
                frame_size = 1280 * 720 * 3
            else:
                frame_size = self.width * self.height * 3
            
            # Read raw bytes for one frame
            raw_frame = self.ffmpeg_process.stdout.read(frame_size)
            
            # If we didn't get a full frame, we're at the end
            if len(raw_frame) < frame_size:
                logger.debug(f"Incomplete frame read ({len(raw_frame)} bytes), end of frames")
                return None
            
            # Convert raw bytes to numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
            
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                logger.info(f"Read frame {self.frame_count} from pipe")
                
            return frame
            
        except Exception as e:
            logger.error(f"Error reading frame from pipe: {str(e)}")
            return None
    
    def close(self):
        """Clean up resources"""
        self.cleanup()
    
    def cleanup(self):
        """Terminate FFmpeg process and remove temporary directory"""
        if self.ffmpeg_process is not None:
            try:
                if self.ffmpeg_process.poll() is None:
                    # Process is still running, terminate it
                    logger.debug("Terminating FFmpeg process")
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error terminating FFmpeg process: {str(e)}")
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
            
            # Close pipes
            try:
                if self.ffmpeg_process.stdout:
                    self.ffmpeg_process.stdout.close()
                if self.ffmpeg_process.stderr:
                    self.ffmpeg_process.stderr.close()
            except Exception as e:
                logger.error(f"Error closing FFmpeg pipes: {str(e)}")
            
            self.ffmpeg_process = None
            
        # Remove temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Removed temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error removing temporary directory: {str(e)}")
            self.temp_dir = None 