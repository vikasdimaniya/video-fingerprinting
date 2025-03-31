import logging
import os
import subprocess
import tempfile
import shutil
import numpy as np
import cv2

from offmark.video.frame_writer import FrameWriter

logger = logging.getLogger(__name__)

class M4sFrameWriter(FrameWriter):
    """Custom FrameWriter for M4S segments that uses intermediate files instead of pipes"""
    
    def __init__(self, output_file, width, height):
        super().__init__()
        logger.info(f"Initializing M4sFrameWriter for {output_file}")
        self.output_file = output_file
        self.width = width
        self.height = height
        self.frames_dir = None
        self.frame_count = 0
        self.fps = 24  # Default fps
        self.closed = False
        
        # Create temp directory for frames
        self.frames_dir = tempfile.mkdtemp(prefix="m4s_frames_")
        logger.debug(f"Created temporary directory for frames: {self.frames_dir}")

    def write(self, frame):
        """Save frame as an image file instead of piping to FFmpeg"""
        if self.closed:
            logger.error("Cannot write to closed writer")
            return
            
        try:
            # Convert frame if needed
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            self.frame_count += 1
            
            # Save frame as image
            frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.png")
            cv2.imwrite(frame_path, frame)
            
            if self.frame_count % 10 == 0:
                logger.debug(f"Saved {self.frame_count} frames to temporary directory")
                
        except Exception as e:
            logger.error(f"Error writing frame: {str(e)}")
            self.close()
            raise

    def close(self):
        """Encode all saved frames to the output M4S file"""
        if self.closed:
            logger.debug("Writer already closed")
            return
            
        self.closed = True
        logger.info(f"Closing M4sFrameWriter after writing {self.frame_count} frames")
        
        if self.frame_count == 0:
            logger.warning("No frames were written, skipping encoding")
            self.__cleanup()
            return
            
        try:
            # Create temporary output file (will be converted to M4S later)
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_mp4 = temp_file.name
                logger.debug(f"Created temporary output file: {temp_mp4}")
            
            # Create FFmpeg command to encode frames to MP4
            cmd = [
                'ffmpeg',
                '-y',
                '-framerate', str(self.fps),
                '-i', os.path.join(self.frames_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-profile:v', 'high',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart+frag_keyframe+empty_moov',
                '-f', 'mp4',
                temp_mp4
            ]
            
            logger.debug(f"Running FFmpeg to encode frames: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg encoding error: {result.stderr}")
                raise RuntimeError(f"Failed to encode frames: {result.stderr}")
                
            # Check if the output is M4S or MP4
            if self.output_file.endswith('.m4s'):
                # For M4S, we need to remove the MOOV atom and keep only the MOOF+MDAT
                # Typically this would require special handling, but for simplicity
                # we'll just rename the file for now
                logger.debug(f"Moving temporary MP4 to output M4S: {self.output_file}")
                shutil.move(temp_mp4, self.output_file)
            else:
                # Just a regular MP4 output
                logger.debug(f"Moving temporary MP4 to output: {self.output_file}")
                shutil.move(temp_mp4, self.output_file)
                
            logger.info(f"Successfully encoded {self.frame_count} frames to {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error encoding frames: {str(e)}")
            if 'temp_mp4' in locals() and os.path.exists(temp_mp4):
                os.unlink(temp_mp4)
            raise
        finally:
            self.__cleanup()
    
    def __cleanup(self):
        """Clean up temporary files"""
        if self.frames_dir and os.path.exists(self.frames_dir):
            try:
                # Remove temporary directory with all frames
                shutil.rmtree(self.frames_dir)
                logger.debug(f"Removed temporary directory: {self.frames_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")
            self.frames_dir = None
            
    def is_alive(self):
        """Check if the writer is available for writing"""
        return not self.closed
    
    def get_last_error(self):
        """Return the last error if any"""
        return None if not self.closed else "Writer is closed"


# Create a debug-enabled version that also saves frames for inspection
class DebugM4sFrameWriter(M4sFrameWriter):
    """Enhanced M4sFrameWriter that also saves frames for debugging"""
    
    def __init__(self, output_file, width, height, debug_dir=None, prefix="frame", save_debug_frames=True):
        super().__init__(output_file, width, height)
        self.debug_dir = debug_dir
        self.prefix = prefix
        self.debug_count = 0
        self.save_debug_frames = save_debug_frames
        
        # Create debug directory if needed
        if self.save_debug_frames and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            logger.info(f"Debug frame writer initialized, saving to {debug_dir}")
        
    def write(self, frame):
        """Write frame to the M4S writer and also save for debugging"""
        # Write to the standard M4sFrameWriter
        super().write(frame)
        
        # Save debug frame (every 5th frame)
        if self.save_debug_frames and self.debug_dir:
            self.debug_count += 1
            if self.debug_count % 5 == 0:
                debug_path = os.path.join(self.debug_dir, f"{self.prefix}_{self.debug_count:04d}.png")
                try:
                    cv2.imwrite(debug_path, frame)
                    logger.debug(f"Saved debug frame {self.debug_count} to {debug_path}")
                except Exception as e:
                    logger.error(f"Error saving debug frame: {str(e)}")
            
    def close(self):
        """Close the M4sFrameWriter"""
        if self.save_debug_frames and self.debug_dir:
            logger.info(f"Debug M4S writer saved {self.debug_count//5} debug frames")
        super().close() 