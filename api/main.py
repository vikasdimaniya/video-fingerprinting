from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import uuid
from pathlib import Path
import logging
from typing import Optional, Dict
import subprocess
import numpy as np
from datetime import datetime
import json
import sys
from pydantic import BaseModel

# Add the tests directory to Python path to import mark_video_to_hls
sys.path.append(str(Path(__file__).parent.parent))
from tests.mark_video_to_hls import (
    segment_video, watermark_segment, convert_segments_to_hls,
    detect_patterns_in_segment, generate_payload_for_segment
)
from tests.detect_watermarks import decode_watermark_pattern
from tests.generate_leak import create_custom_hls_playlist, concatenate_segments

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Watermarking Viewer")

# Define base directories
BASE_DIR = Path(__file__).parent.parent
VIDEO_DIR = BASE_DIR / "video_content"
PROCESSED_DIR = VIDEO_DIR / "processed"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Create necessary directories
VIDEO_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
(PROCESSED_DIR / "hls").mkdir(exist_ok=True)

# Initialize templates and static files
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/hls", StaticFiles(directory=str(PROCESSED_DIR / "hls")), name="hls")

# Global counter for unique watermark patterns
watermark_counter = 0

# Store view history
view_history: Dict[str, dict] = {}

def generate_unique_payload():
    """Generate a unique 8-bit payload for watermarking"""
    global watermark_counter
    # Generate a random 8-bit pattern instead of using counter
    payload = np.random.randint(0, 2, size=8)
    watermark_counter += 1
    return payload

def process_video_to_hls(input_path: Path, output_dir: Path, base_payload: np.ndarray, num_copies: int = 3):
    """Process video with watermark and create all possible versions"""
    try:
        # Clean up existing directories if they exist
        segments_dir = output_dir / "segments"
        marked_segments_dir = output_dir / "marked_segments"
        hls_dir = output_dir / "hls"
        
        # Remove existing directories if they exist
        for directory in [segments_dir, marked_segments_dir]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(exist_ok=True)
        
        # Create HLS directory if it doesn't exist
        hls_dir.mkdir(exist_ok=True)
        
        # Clean up only segment files in HLS directory
        for file in hls_dir.glob("*.m4s"):
            file.unlink()
        
        # Step 1: Segment the video
        logger.info("Step 1: Segmenting video")
        segments = segment_video(
            str(input_path),
            str(segments_dir / "segment_%03d.mp4"),
            segment_duration=2
        )
        
        # Step 2: Watermark each segment with multiple versions
        logger.info("Step 2: Watermarking segments")
        successful_segments = {}  # Track which segments were successfully marked
        segment_copies_info = {"segments": {}}  # For segment_copies.json
        
        for i, segment in enumerate(segments):
            segment_copies_info["segments"][str(i)] = []  # Initialize list for this segment
            try:
                # Create num_copies watermarked versions of each segment
                for copy_index in range(num_copies):
                    # First watermark to a temporary MP4
                    temp_output = marked_segments_dir / f"marked_seg{i:03d}_copy{copy_index}.mp4"
                    segment_payload = generate_payload_for_segment(i, copy_index)
                    
                    watermark_info = watermark_segment(
                        segment,
                        str(temp_output),
                        segment_number=i,
                        copy_index=copy_index
                    )
                    
                    # Convert to HLS segment format
                    output_file = hls_dir / f"marked_seg{i:03d}_copy{copy_index}.m4s"
                    cmd = [
                        'ffmpeg',
                        '-i', str(temp_output),
                        '-c:v', 'copy',
                        '-c:a', 'copy',
                        '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
                        '-f', 'mp4',
                        str(output_file)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    logger.info(f"Created segment {output_file}")
                    
                    # Store info about this copy
                    successful_segments[f"marked_seg{i:03d}_copy{copy_index}.m4s"] = {
                        "segment_number": i,
                        "copy_index": copy_index,
                        "payload": segment_payload.tolist(),
                        "file_path": str(output_file)
                    }
                    
                    # Add to segment_copies_info - use .m4s extension
                    segment_copies_info["segments"][str(i)].append({
                        "file": f"marked_seg{i:03d}_copy{copy_index}.m4s",
                        "payload": segment_payload.tolist(),
                        "copy_index": copy_index
                    })
                
            except Exception as e:
                logger.warning(f"Failed to watermark segment {i}: {str(e)}")
                # Copy original segment as fallback
                output_file = hls_dir / f"marked_seg{i:03d}_copy0.m4s"
                cmd = [
                    'ffmpeg',
                    '-i', segment,
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
                    '-f', 'mp4',
                    str(output_file)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"Created fallback segment {output_file}")
                
                # Add fallback to segment_copies_info - use .m4s extension
                segment_copies_info["segments"][str(i)].append({
                    "file": f"marked_seg{i:03d}_copy0.m4s",
                    "payload": generate_payload_for_segment(i, 0).tolist(),
                    "copy_index": 0
                })
        
        # Save segment mapping information
        mapping_file = output_dir / "segment_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump({
                "successful_segments": successful_segments,
                "num_copies": num_copies,
                "description": "Maps segment numbers to their watermarked versions"
            }, f, indent=2)
            
        # Save segment copies information
        copies_file = output_dir / "segment_copies.json"
        with open(copies_file, 'w') as f:
            json.dump(segment_copies_info, f, indent=2)
        logger.info(f"Created segment copies file at {copies_file}")
            
        # Create a base playlist.m3u8 file that will be used as a template
        playlist_content = "#EXTM3U\n"
        playlist_content += "#EXT-X-VERSION:7\n"
        playlist_content += "#EXT-X-TARGETDURATION:2\n"
        playlist_content += "#EXT-X-MEDIA-SEQUENCE:0\n\n"
        
        # Add all segments for copy 0 as the base playlist
        for i in range(len(segments)):
            segment_file = f"marked_seg{i:03d}_copy0.m4s"
            if (hls_dir / segment_file).exists():
                playlist_content += f"#EXTINF:2.0,\n"
                playlist_content += f"{segment_file}\n"
        
        playlist_content += "#EXT-X-ENDLIST\n"
        
        # Save the base playlist
        playlist_path = hls_dir / "playlist.m3u8"
        with open(playlist_path, 'w') as f:
            f.write(playlist_content)
        logger.info(f"Created base playlist at {playlist_path}")
            
        # Create a simple master playlist
        master_content = "#EXTM3U\n"
        master_content += "#EXT-X-VERSION:7\n"
        master_content += "#EXT-X-STREAM-INF:BANDWIDTH=2000000\n"
        master_content += "playlist.m3u8\n"
        
        master_path = hls_dir / "master.m3u8"
        with open(master_path, 'w') as f:
            f.write(master_content)
        logger.info(f"Created master playlist at {master_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

def create_view_playlist(view_number: int, num_copies: int, num_segments: int) -> str:
    """Create an m3u8 playlist for a specific view number"""
    logger.info(f"Creating playlist for view {view_number} with {num_segments} segments and {num_copies} copies")
    
    # Convert view number to base-num_copies
    view_base = []
    temp_view = view_number
    while temp_view > 0:
        view_base.append(temp_view % num_copies)
        temp_view //= num_copies
    # Pad with zeros if needed
    while len(view_base) < num_segments:
        view_base.append(0)
    # Reverse to get correct order
    view_base.reverse()
    
    logger.info(f"View base-{num_copies} representation: {view_base}")
    
    # Create m3u8 content with fMP4 specific tags
    m3u8_content = "#EXTM3U\n"
    m3u8_content += "#EXT-X-VERSION:7\n"
    m3u8_content += "#EXT-X-TARGETDURATION:2\n"
    m3u8_content += "#EXT-X-MEDIA-SEQUENCE:0\n"
    m3u8_content += "#EXT-X-MAP:URI=\"/hls/init.mp4\"\n\n"  # Add initialization segment
    
    # Add segments based on view_base
    for i, copy_index in enumerate(view_base):
        segment_file = f"marked_seg{i:03d}_copy{copy_index}.m4s"
        # Check if the segment file exists
        if (PROCESSED_DIR / "hls" / segment_file).exists():
            m3u8_content += f"#EXTINF:2.0,\n"
            m3u8_content += f"/hls/{segment_file}\n\n"
            logger.info(f"Added segment {segment_file} to playlist")
        else:
            logger.warning(f"Segment file {segment_file} not found")
    
    m3u8_content += "#EXT-X-ENDLIST\n"
    return m3u8_content

@app.get("/")
async def home():
    """Redirect to upload page"""
    return RedirectResponse(url="/upload")

@app.get("/upload")
async def upload_page(request: Request):
    """Show the upload page"""
    return templates.TemplateResponse(
        "upload.html",
        {"request": request}
    )

@app.post("/upload")
async def upload_video(file: UploadFile):
    """Upload a video file and process it with all possible watermarked versions"""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = VIDEO_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / "source.mp4"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process video with all watermarked versions
        success = process_video_to_hls(
            file_path,
            PROCESSED_DIR,
            generate_unique_payload(),
            num_copies=3  # Create 3 versions of each segment
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process video")
        
        # Redirect to view page after successful upload
        return RedirectResponse(url="/view", status_code=303)
        
    except Exception as e:
        # Clean up uploaded file if processing fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-view")
async def start_view(view_data: dict):
    """Initialize a new view with a unique combination of watermarked segments"""
    try:
        username = view_data.get("username")
        num_copies = view_data.get("num_copies", 3)  # Default to 3 copies if not specified
        
        if not username:
            raise HTTPException(status_code=400, detail="Username is required")
        
        logger.info(f"Starting new view for user: {username} with {num_copies} copies")
        
        # Get current view count to determine view number
        if (PROCESSED_DIR / "view_history.json").exists():
            with open(PROCESSED_DIR / "view_history.json", 'r') as f:
                view_history = json.load(f)
                view_number = len(view_history)
        else:
            view_history = {}
            view_number = 0
        
        # Load segment mapping to get number of segments and copies
        if not (PROCESSED_DIR / "segment_mapping.json").exists():
            raise HTTPException(status_code=404, detail="No processed video found. Please upload a video first.")
            
        segment_mapping = json.loads((PROCESSED_DIR / "segment_mapping.json").read_text())
        num_segments = max(
            info["segment_number"] for info in segment_mapping["successful_segments"].values()
        ) + 1
        
        logger.info(f"Found {num_segments} segments with {num_copies} copies each")
        
        # Create HLS directory for this view
        hls_dir = PROCESSED_DIR / "hls"
        hls_dir.mkdir(exist_ok=True)
        
        # Create playlist for this view
        playlist_content = create_view_playlist(view_number, num_copies, num_segments)
        playlist_file = hls_dir / "master.m3u8"
        with open(playlist_file, 'w') as f:
            f.write(playlist_content)
        
        # Convert view number to base-num_copies to determine which copy of each segment to use
        view_base = []
        temp_view = view_number
        while temp_view > 0:
            view_base.append(temp_view % num_copies)
            temp_view //= num_copies
        # Pad with zeros if needed
        while len(view_base) < num_segments:
            view_base.append(0)
        # Reverse to get correct order
        view_base.reverse()
        
        # Get the segment patterns for this view using the correct copy index for each segment
        segment_patterns = {}
        for i, copy_index in enumerate(view_base):
            segment_key = f"marked_seg{i:03d}_copy{copy_index}.m4s"
            if segment_key in segment_mapping["successful_segments"]:
                segment_patterns[segment_key] = segment_mapping["successful_segments"][segment_key]
        
        # Update view history
        view_id = str(uuid.uuid4())
        view_history[view_id] = {
            "username": username,
            "timestamp": datetime.now().isoformat(),
            "view_number": view_number,
            "num_copies": num_copies,
            "num_segments": num_segments,
            "segment_patterns": segment_patterns,
            "segment_mapping": {
                "successful_segments": segment_patterns,
                "num_copies": num_copies,
                "description": "Maps segment numbers to their watermarked versions"
            }
        }
        
        # Save updated view history
        with open(PROCESSED_DIR / "view_history.json", 'w') as f:
            json.dump(view_history, f, indent=2)
        
        logger.info(f"Successfully created view {view_number} for user {username}")
        
        return {
            "status": "success",
            "view_id": view_id,
            "view_number": view_number,
            "num_copies": num_copies,
            "num_segments": num_segments,
            "segment_patterns": segment_patterns
        }
        
    except HTTPException as he:
        logger.error(f"HTTP error in start-view: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in start-view: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start view: {str(e)}")

@app.get("/view")
async def view_video(request: Request):
    """Show video viewer page"""
    source_video = VIDEO_DIR / "uploads" / "source.mp4"
    if not source_video.exists():
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Source video not found. Please upload first."}
        )
    
    # Get the most recent view's watermark patterns
    if (PROCESSED_DIR / "view_history.json").exists():
        with open(PROCESSED_DIR / "view_history.json", 'r') as f:
            view_history = json.load(f)
            if view_history:
                # Get the most recent view
                latest_view = max(view_history.items(), key=lambda x: x[1]["timestamp"])
                # Get the segment patterns for this view
                segment_patterns = latest_view[1].get("segment_patterns", {})
            else:
                segment_patterns = {}
    else:
        segment_patterns = {}
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "segment_patterns": segment_patterns
        }
    )

@app.get("/stream")
async def stream_video():
    """Stream the watermarked video"""
    processed_dir = PROCESSED_DIR / "source"
    if not processed_dir.exists():
        raise HTTPException(status_code=404, detail="No processed video found")
    
    master_playlist = processed_dir / "hls" / "master.m3u8"
    media_playlist = processed_dir / "hls" / "playlist.m3u8"
    
    if not master_playlist.exists():
        raise HTTPException(status_code=404, detail="Master playlist not found")
    
    # Read the master playlist content
    with open(master_playlist, 'r') as f:
        master_content = f.read()
    
    # Read the media playlist content
    with open(media_playlist, 'r') as f:
        media_content = f.read()
    
    # Return the appropriate playlist based on the Accept header
    accept_header = master_content if "master" in master_content.lower() else media_content
    
    # Return the playlist content with correct content type
    return Response(content=accept_header, media_type="application/x-mpegURL")

@app.get("/download/{username}")
async def download_video(username: str):
    """Download the watermarked video for a specific user"""
    try:
        # Get the most recent processed video
        processed_dir = PROCESSED_DIR / "source"
        if not processed_dir.exists():
            raise HTTPException(status_code=404, detail="No processed video found")
        
        # Load view history to find user's watermarked segments
        if not (PROCESSED_DIR / "view_history.json").exists():
            raise HTTPException(status_code=404, detail="No view history found")
            
        with open(PROCESSED_DIR / "view_history.json", 'r') as f:
            view_history = json.load(f)
        
        # Find the user's view entry
        user_view = None
        for view_id, view_data in view_history.items():
            if view_data.get("username") == username:
                user_view = view_data
                break
        
        if not user_view:
            raise HTTPException(status_code=404, detail=f"No view history found for user {username}")
        
        # Create temporary directory for download
        temp_dir = Path("temp_download")
        temp_dir.mkdir(exist_ok=True)
        
        # Create output file path
        output_file = temp_dir / "downloaded_video.mp4"
        
        # Get the segments mapping for this user's view
        segment_mapping = user_view["segment_mapping"]
        hls_dir = processed_dir / "hls"
        
        # Create a temporary directory for segments
        temp_segments_dir = temp_dir / "segments"
        temp_segments_dir.mkdir(exist_ok=True)
        
        # Copy all necessary files to temp directory
        for file in hls_dir.glob("*"):
            shutil.copy2(file, temp_segments_dir)
        
        # Create a temporary playlist with relative paths
        temp_playlist = temp_segments_dir / "playlist.m3u8"
        with open(hls_dir / "playlist.m3u8", 'r') as f:
            playlist_content = f.read()
        
        # Write the playlist
        with open(temp_playlist, 'w') as f:
            f.write(playlist_content)
        
        # Use ffmpeg to convert HLS to MP4
        cmd = [
            'ffmpeg',
            '-allowed_extensions', 'ALL',
            '-i', str(temp_playlist),
            '-c', 'copy',
            '-movflags', '+faststart',
            str(output_file)
        ]
        
        # Run ffmpeg command
        subprocess.run(cmd, check=True)
        
        # Return the concatenated file
        response = FileResponse(
            output_file,
            media_type="video/mp4",
            filename=f"watermarked_video_{username}.mp4"
        )
        
        # Clean up temp directory after sending file
        def cleanup():
            shutil.rmtree(temp_dir)
        
        response.background = cleanup
        return response
        
    except Exception as e:
        # Clean up on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/view-history")
async def get_view_history():
    """Get the view history (for detection script)"""
    return view_history

@app.post("/detect")
async def detect_leak(file: UploadFile):
    """Detect which copy was leaked based on the watermark"""
    try:
        # Create temporary directory for detection
        temp_dir = PROCESSED_DIR / "temp_detection"
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file for detection
        leaked_file = temp_dir / "leaked.mp4"
        with open(leaked_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load view history to get successful segments info
        if not (PROCESSED_DIR / "view_history.json").exists():
            return {"error": "No view history found"}
            
        with open(PROCESSED_DIR / "view_history.json", 'r') as f:
            view_history = json.load(f)
        
        # First, check if this segment was successfully marked in any view
        any_successful = False
        for view_id, view_data in view_history.items():
            segment_mapping = view_data.get("segment_mapping", {})
            successful_segments = segment_mapping.get("successful_segments", {})
            if successful_segments:
                any_successful = True
                break
        
        if not any_successful:
            return {"error": "No successfully marked segments found in view history"}
        
        # Detect watermark in the leaked segment
        pattern, frequency, success = detect_patterns_in_segment(str(leaked_file))
        
        if not success or pattern is None:
            # Try to determine if this is a segment that failed to embed
            segment_number = None
            for view_id, view_data in view_history.items():
                segment_mapping = view_data.get("segment_mapping", {})
                successful_segments = segment_mapping.get("successful_segments", {})
                if not successful_segments:
                    continue
                
                # Get the number of segments from the view data
                num_segments = view_data.get("num_segments", 0)
                
                # Check each segment number
                for i in range(num_segments):
                    # Check all possible copies for this segment
                    segment_found = False
                    for copy_index in range(view_data.get("num_copies", 3)):
                        segment_key = f"marked_seg{i:03d}_copy{copy_index}.m4s"
                        if segment_key in successful_segments:
                            segment_found = True
                            break
                    
                    if not segment_found:
                        segment_number = i
                        logger.info(f"Found failed segment: {segment_number} (not found in any copy)")
                        break
                if segment_number is not None:
                    break
            
            if segment_number is not None:
                return {
                    "status": "no_match",
                    "error": f"Segment {segment_number} was not successfully marked in any view",
                    "segment_number": segment_number,
                    "note": "This segment failed to embed during watermarking"
                }
            return {"error": "Could not detect watermark pattern"}
        
        # Decode the detected pattern
        segment_number, copy_index = decode_watermark_pattern(pattern)
        
        if segment_number is None:
            return {"error": "Could not decode watermark pattern"}
        
        # Find matching view in history
        matches = []
        for view_id, view_data in view_history.items():
            # Check if this segment was successfully marked for this view
            segment_mapping = view_data.get("segment_mapping", {})
            successful_segments = segment_mapping.get("successful_segments", {})
            
            # Skip if we don't have successful segments info
            if not successful_segments:
                continue
                
            # Check if the detected segment was successfully marked
            segment_key = f"marked_seg{segment_number:03d}_copy{copy_index}.m4s"
            if segment_key not in successful_segments:
                logger.info(f"Skipping segment {segment_number} as it wasn't successfully marked")
                continue
            
            # Compare patterns for this specific segment
            segment_info = successful_segments[segment_key]
            if np.array_equal(pattern, np.array(segment_info["payload"])):
                matches.append({
                    "view_id": view_id,
                    "username": view_data.get("username", "Unknown"),
                    "timestamp": view_data["timestamp"],
                    "payload": segment_info["payload"],
                    "segment_number": segment_number,
                    "copy_index": copy_index,
                    "frequency": float(frequency)
                })
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        if matches:
            return {
                "status": "success",
                "matches": matches,
                "detected_payload": pattern.tolist(),
                "segment_number": segment_number,
                "copy_index": copy_index,
                "frequency": float(frequency)
            }
        else:
            # If we got here and have no matches, it means either:
            # 1. The segment wasn't successfully marked
            # 2. The pattern doesn't match any known views
            segment_key = f"marked_seg{segment_number:03d}_copy{copy_index}.m4s"
            was_marked = any(
                segment_key in view_data.get("segment_mapping", {}).get("successful_segments", {})
                for view_data in view_history.values()
            )
            
            if not was_marked:
                return {
                    "status": "no_match",
                    "error": f"Segment {segment_number} was not successfully marked in any view",
                    "segment_number": segment_number,
                    "note": "This segment failed to embed during watermarking"
                }
            else:
                return {
                    "status": "no_match",
                    "detected_payload": pattern.tolist() if pattern is not None else None,
                    "segment_number": segment_number,
                    "copy_index": copy_index,
                    "frequency": float(frequency),
                    "note": "Pattern detected but doesn't match any known views"
                }
            
    except Exception as e:
        # Clean up on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return {"error": str(e)}

@app.get("/detect")
async def detect_page(request: Request):
    """Show the detection page"""
    return templates.TemplateResponse(
        "detect.html",
        {"request": request}
    )

@app.get("/view/{view_id}")
async def get_video(view_id: str):
    """Get the HLS stream for a specific view"""
    try:
        logger.info(f"Getting video for view ID: {view_id}")
        
        # Load view history
        if not (PROCESSED_DIR / "view_history.json").exists():
            logger.error("View history file not found")
            raise HTTPException(status_code=404, detail="View not found")
            
        with open(PROCESSED_DIR / "view_history.json", 'r') as f:
            view_history = json.load(f)
            
        if view_id not in view_history:
            logger.error(f"View ID {view_id} not found in history")
            raise HTTPException(status_code=404, detail="View not found")
            
        view_info = view_history[view_id]
        
        # Convert view number to pattern string
        view_base = []
        temp_view = view_info["view_number"]
        num_copies = view_info["num_copies"]
        while temp_view > 0:
            view_base.append(temp_view % num_copies)
            temp_view //= num_copies
        # Pad with zeros if needed
        while len(view_base) < view_info["num_segments"]:
            view_base.append(0)
        # Reverse to get correct order
        view_base.reverse()
        
        # Create HLS playlist content
        playlist_content = "#EXTM3U\n"
        playlist_content += "#EXT-X-VERSION:7\n"
        playlist_content += "#EXT-X-TARGETDURATION:2\n"
        playlist_content += "#EXT-X-MEDIA-SEQUENCE:0\n\n"
        
        # Add segments based on view_base
        for i, copy_index in enumerate(view_base):
            segment_file = f"marked_seg{i:03d}_copy{copy_index}.m4s"
            # Check if the segment file exists
            if (PROCESSED_DIR / "hls" / segment_file).exists():
                playlist_content += f"#EXTINF:2.0,\n"
                playlist_content += f"/hls/{segment_file}\n"
                logger.info(f"Added segment {segment_file} to playlist")
            else:
                logger.warning(f"Segment file {segment_file} not found")
        
        playlist_content += "#EXT-X-ENDLIST\n"
        
        logger.info("Created playlist content")
        logger.debug(f"Playlist content:\n{playlist_content}")
            
        # Return the playlist content with correct content type and CORS headers
        return Response(
            content=playlist_content,
            media_type="application/x-mpegURL",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control": "no-cache"  # Prevent caching of the playlist
            }
        )
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hls/{filename}")
async def get_hls_segment(filename: str):
    """Get an HLS segment file"""
    try:
        logger.info(f"Requested segment: {filename}")
        segment_path = PROCESSED_DIR / "hls" / filename
        
        if not segment_path.exists():
            logger.error(f"Segment not found at {segment_path}")
            raise HTTPException(status_code=404, detail="Segment not found")
            
        logger.info(f"Serving segment from {segment_path}")
        return FileResponse(
            segment_path,
            media_type="video/iso.segment",  # Correct MIME type for fMP4 segments
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control": "no-cache",  # Prevent caching of segments
                "Content-Type": "video/iso.segment"  # Explicitly set content type
            }
        )
    except Exception as e:
        logger.error(f"Error serving segment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-view/{view_id}")
async def download_view(view_id: str):
    try:
        logger.info(f"Starting download for view ID: {view_id}")
        
        # Load view history
        with open(PROCESSED_DIR / "view_history.json", 'r') as f:
            view_history = json.load(f)
        
        if view_id not in view_history:
            logger.error(f"View ID {view_id} not found in history")
            raise HTTPException(status_code=404, detail="View not found")
        
        view_info = view_history[view_id]
        view_number = view_info['view_number']
        num_copies = view_info['num_copies']
        num_segments = view_info['num_segments']
        
        logger.info(f"View info: number={view_number}, copies={num_copies}, segments={num_segments}")
        
        # Create view_base array
        view_base = []
        temp_view = view_number
        while temp_view > 0:
            view_base.append(temp_view % num_copies)
            temp_view //= num_copies
        # Pad with zeros if needed
        while len(view_base) < num_segments:
            view_base.append(0)
        # Reverse to get correct order
        view_base.reverse()
        
        logger.info(f"Generated view_base array: {view_base}")
        
        # Create temp directory if it doesn't exist
        temp_dir = PROCESSED_DIR / "temp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Create output file path
        output_file = temp_dir / f"view_{view_id}.mp4"
        logger.info(f"Output file path: {output_file}")
        
        # Log HLS directory contents
        hls_dir = PROCESSED_DIR / "hls"
        logger.info(f"HLS directory contents: {list(hls_dir.glob('*.m4s'))}")
        
        # Create list of segment files based on view_base
        segment_files = []
        for i, copy_index in enumerate(view_base):
            segment_file = f"marked_seg{i:03d}_copy{copy_index}.m4s"
            segment_path = hls_dir / segment_file
            if segment_path.exists():
                segment_files.append(str(segment_path))
                logger.info(f"Added segment file: {segment_path}")
            else:
                logger.warning(f"Segment file not found: {segment_path}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Segment file not found: {segment_file}"
                )
        
        if not segment_files:
            raise HTTPException(
                status_code=404,
                detail="No segment files found"
            )
        
        logger.info(f"Concatenating {len(segment_files)} segments into {output_file}")
        
        # Use concatenate_segments to create the MP4
        try:
            concatenate_segments(segment_files, str(output_file))
            logger.info(f"Successfully created MP4 file at {output_file}")
        except Exception as e:
            logger.error(f"Error concatenating segments: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to concatenate segments: {str(e)}"
            )
        
        if not output_file.exists():
            logger.error(f"Output file was not created at {output_file}")
            raise HTTPException(
                status_code=500,
                detail="Output file was not created"
            )
        
        # Get file size
        file_size = output_file.stat().st_size
        logger.info(f"Output file size: {file_size} bytes")
        
        # Create response with cleanup
        async def cleanup():
            try:
                if output_file.exists():
                    output_file.unlink()
                    logger.info("Cleaned up output file")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        logger.info("Returning StreamingResponse")
        return StreamingResponse(
            open(output_file, 'rb'),
            media_type='video/mp4',
            headers={
                'Content-Disposition': f'attachment; filename="watermarked_video_{view_id}.mp4"',
                'Content-Length': str(file_size)
            },
            background=cleanup
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        logger.exception("Full traceback:")  # This will log the full stack trace
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading video: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 