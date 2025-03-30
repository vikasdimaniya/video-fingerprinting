import logging
import os
import subprocess
import tempfile
import numpy as np

from offmark.video.frame_reader import FrameReader

logger = logging.getLogger(__name__)

class M4sFrameReader(FrameReader):
    """FrameReader implementation for M4S files that works with init segments"""
    
    def __init__(self, m4s_file, init_segment):
        super().__init__()
        logger.info(f"Initializing M4sFrameReader for {m4s_file}")
        self.m4s_file = m4s_file
        self.init_segment = init_segment
        self.temp_mp4 = None
        self.ffmpeg = None
        self.width = None
        self.height = None
        self.frame_count = 0
        
        self.__prepare_mp4()
        self.__get_dimensions()
        self.__start_ffmpeg()
        
    def __prepare_mp4(self):
        """Create a temporary MP4 by concatenating init segment and M4S segment"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            self.temp_mp4 = temp_file.name
            logger.debug(f"Created temporary MP4 file: {self.temp_mp4}")
        
        # Simple binary concatenation
        with open(self.temp_mp4, 'wb') as outfile:
            # Copy init segment content
            with open(self.init_segment, 'rb') as infile:
                init_content = infile.read()
                outfile.write(init_content)
                logger.debug(f"Copied {len(init_content)} bytes from init segment")
            
            # Copy m4s segment content
            with open(self.m4s_file, 'rb') as infile:
                m4s_content = infile.read()
                outfile.write(m4s_content)
                logger.debug(f"Copied {len(m4s_content)} bytes from m4s segment")
        
        logger.debug(f"Created combined file with size: {os.path.getsize(self.temp_mp4)} bytes")
        
    def __get_dimensions(self):
        """Get video dimensions using ffprobe"""
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            self.temp_mp4
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            dimensions = result.stdout.strip().split(',')
            if len(dimensions) == 2:
                self.width = int(dimensions[0])
                self.height = int(dimensions[1])
                logger.info(f"Video dimensions: {self.width}x{self.height}")
            else:
                # If ffprobe fails, use default dimensions
                self.width = 1280
                self.height = 720
                logger.warning(f"Could not determine dimensions, using defaults: {self.width}x{self.height}")
        except Exception as e:
            # If ffprobe fails, use default dimensions
            self.width = 1280
            self.height = 720
            logger.warning(f"Error getting dimensions: {str(e)}, using defaults: {self.width}x{self.height}")
    
    def __start_ffmpeg(self):
        """Start FFmpeg process to decode frames"""
        # Frame size in bytes (RGB24)
        self.frame_size_bytes = self.width * self.height * 3
        
        # FFmpeg command with many options to improve robustness
        cmd = [
            'ffmpeg',
            '-y',
            '-analyzeduration', '100000000',  # 100 seconds
            '-probesize', '100000000',        # 100MB
            '-i', self.temp_mp4,
            '-vsync', '0',                    # Disable frame dropping
            '-f', 'rawvideo',                 # Output format
            '-pix_fmt', 'rgb24',              # Output pixel format 
            '-v', 'warning',                  # Verbosity level
            'pipe:'                           # Output to pipe
        ]
        
        logger.debug(f"Starting FFmpeg with command: {' '.join(cmd)}")
        
        try:
            self.ffmpeg = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.debug("FFmpeg process started")
        except Exception as e:
            logger.error(f"Error starting FFmpeg: {str(e)}")
            self.cleanup()
            raise

    def read(self):
        """Read one frame in RGB format - implements FrameReader interface"""
        if not self.ffmpeg:
            logger.error("FFmpeg process not available")
            return None
        
        try:
            # Read raw frame data
            frame_bytes = self.ffmpeg.stdout.read(self.frame_size_bytes)
            
            # Check if we reached the end
            if len(frame_bytes) == 0:
                logger.debug("End of stream reached")
                return None
            
            # Check if we got a complete frame
            if len(frame_bytes) != self.frame_size_bytes:
                logger.warning(f"Incomplete frame: got {len(frame_bytes)} bytes, expected {self.frame_size_bytes}")
                return None
            
            # Convert to numpy array
            frame = np.frombuffer(frame_bytes, np.uint8).reshape(self.height, self.width, 3)
            
            # Increment frame counter
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                logger.debug(f"Read frame {self.frame_count}")
            
            return frame
                
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return None
    
    def close(self):
        """Clean up resources - implements FrameReader interface"""
        logger.debug("Cleaning up resources")
        
        # Terminate FFmpeg process
        if self.ffmpeg:
            try:
                logger.debug("Terminating FFmpeg process")
                self.ffmpeg.terminate()
                try:
                    self.ffmpeg.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg did not terminate gracefully, killing")
                    self.ffmpeg.kill()
            except Exception as e:
                logger.error(f"Error terminating FFmpeg: {str(e)}")
            self.ffmpeg = None
        
        # Remove temporary file
        if self.temp_mp4 and os.path.exists(self.temp_mp4):
            try:
                os.unlink(self.temp_mp4)
                logger.debug(f"Removed temporary file: {self.temp_mp4}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")
            self.temp_mp4 = None 