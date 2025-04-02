#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import tempfile
import random
import logging
import re
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

def load_segment_copies(copies_file):
    """
    Load segment copies information from segment_copies.json
    
    Args:
        copies_file: Path to segment_copies.json file
        
    Returns:
        dict: Segment copies information
    """
    if not os.path.exists(copies_file):
        raise FileNotFoundError(f"Copies file {copies_file} not found")
    
    with open(copies_file, 'r') as f:
        segment_copies = json.load(f)
    
    return segment_copies

def load_segment_mapping(base_dir):
    """
    Load segment mapping information from segment_mapping.json
    
    Args:
        base_dir: Base directory containing the segment_mapping.json file
        
    Returns:
        dict: Mapping from HLS m4s files to their source watermarked segment files
    """
    mapping_file = os.path.join(base_dir, 'segment_mapping.json')
    if not os.path.exists(mapping_file):
        logger.warning(f"Mapping file {mapping_file} not found. HLS playlist creation may not be accurate.")
        return {}
    
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Return the mapping dictionary
    return mapping_data.get("hls_to_watermarked", {})

def select_copies(segment_copies_info, copies_file_path, pattern=None, random_seed=None):
    """
    Select one copy of each segment based on pattern or randomly
    
    Args:
        segment_copies_info: Segment copies information from segment_copies.json
        copies_file_path: Path to the copies file (used for resolving relative paths)
        pattern: Optional pattern of copy indexes to use (e.g. "0123" means copy 0 for segment 0, copy 1 for segment 1, etc.)
        random_seed: Random seed for reproducibility when generating random patterns
        
    Returns:
        list: List of selected segment files with full paths
        list: Copy indexes used for each segment
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    segments_info = segment_copies_info["segments"]
    sorted_segment_numbers = sorted([int(s) for s in segments_info.keys()])
    
    selected_files = []
    copy_pattern = []
    
    # If pattern is provided, use it to select copies
    if pattern:
        if len(pattern) < len(sorted_segment_numbers):
            raise ValueError(f"Pattern '{pattern}' is too short for {len(sorted_segment_numbers)} segments")
        
        for i, segment_number in enumerate(sorted_segment_numbers):
            segment_key = str(segment_number)
            segment_copies = segments_info[segment_key]
            copy_index = int(pattern[i]) % len(segment_copies)
            copy_pattern.append(copy_index)
            
            selected_copy = segment_copies[copy_index]
            file_path = os.path.join(os.path.dirname(copies_file_path), "marked_segments", selected_copy["file"])
            selected_files.append(file_path)
    else:
        # Select random copies
        for segment_number in sorted_segment_numbers:
            segment_key = str(segment_number)
            segment_copies = segments_info[segment_key]
            copy_index = random.randint(0, len(segment_copies) - 1)
            copy_pattern.append(copy_index)
            
            selected_copy = segment_copies[copy_index]
            file_path = os.path.join(os.path.dirname(copies_file_path), "marked_segments", selected_copy["file"])
            selected_files.append(file_path)
    
    return selected_files, copy_pattern

def concatenate_segments(segment_files, output_file):
    """
    Concatenate multiple segment files into a single MP4 file
    
    Args:
        segment_files: List of segment files to concatenate
        output_file: Path to output MP4 file
    """
    # Create a temp file for the concat list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        # Write file paths in ffmpeg concat format
        for segment_file in segment_files:
            f.write(f"file '{os.path.abspath(segment_file)}'\n")
    
    # Concatenate segments
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',  # Use stream copy to avoid reencoding
        output_file
    ]
    
    logger.info(f"Concatenating {len(segment_files)} segments into {output_file}")
    subprocess.run(cmd, check=True)
    
    # Clean up the temp file
    os.unlink(concat_file)
    
    return output_file

def extract_segment_number(filename):
    """
    Extract segment number from various filename patterns
    
    Args:
        filename: Filename to parse (basename or full path)
        
    Returns:
        int: Segment number or None if not found
    """
    # Extract just the basename
    basename = os.path.basename(filename)
    
    # Try different patterns
    
    # Pattern 1: marked_seg{X}_copy{Y}.mp4
    match = re.search(r'seg(\d+)_copy', basename)
    if match:
        return int(match.group(1))
    
    # Pattern 2: segment_{XXX}.mp4 or segment_{XXX}.m4s
    match = re.search(r'segment_0*(\d+)', basename)
    if match:
        return int(match.group(1))
    
    # Pattern 3: marked_segment_{XXX}.mp4
    match = re.search(r'marked_segment_0*(\d+)', basename)
    if match:
        return int(match.group(1))
    
    # Pattern 4: Any number in the filename (last resort)
    matches = re.findall(r'(\d+)', basename)
    if matches:
        return int(matches[0])
    
    return None

def extract_copy_index(filename):
    """
    Extract copy index from a watermarked segment filename
    
    Args:
        filename: Filename like marked_seg{X}_copy{Y}.mp4
        
    Returns:
        int: Copy index or None if not found
    """
    match = re.search(r'copy(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def create_custom_hls_playlist(base_dir, pattern_str, selected_segment_files=None):
    """
    Create a custom HLS playlist (m3u8) that references existing m4s segments
    without re-encoding anything. This allows for browser playback with the
    specifically selected segments.
    
    Args:
        base_dir: Base directory containing the HLS files
        pattern_str: The pattern string to use as an identifier
        selected_segment_files: Optional list of segment files (for naming reference)
        
    Returns:
        str: Path to the generated custom playlist
    """
    # Locate the original HLS directory and master playlist
    hls_dir = os.path.join(base_dir, "hls")
    if not os.path.exists(hls_dir):
        raise FileNotFoundError(f"HLS directory not found at {hls_dir}")
    
    master_playlist_path = os.path.join(hls_dir, "master.m3u8")
    if not os.path.exists(master_playlist_path):
        raise FileNotFoundError(f"Master playlist not found at {master_playlist_path}")
    
    # Read the original playlist.m3u8 to get structure and headers
    original_playlist_path = os.path.join(hls_dir, "playlist.m3u8")
    if not os.path.exists(original_playlist_path):
        raise FileNotFoundError(f"Original playlist not found at {original_playlist_path}")
    
    with open(original_playlist_path, 'r') as f:
        original_playlist_content = f.read()
    
    # Extract header (everything before the first segment reference)
    header_lines = []
    segment_lines = []
    in_header = True
    
    for line in original_playlist_content.split('\n'):
        if line.strip().startswith('#EXTINF') or line.strip().endswith('.m4s'):
            in_header = False
        
        if in_header:
            header_lines.append(line)
        else:
            segment_lines.append(line)
    
    # Load segment mapping
    segment_mapping = load_segment_mapping(base_dir)
    logger.debug(f"Loaded segment mapping: {segment_mapping}")
    
    if not segment_mapping and selected_segment_files:
        logger.warning("No segment mapping found. Using filename-based matching as fallback.")
    
    # Extract segment information from the playlist
    segments_info = []
    current_extinf = None
    
    for line in segment_lines:
        if line.startswith('#EXTINF:'):
            current_extinf = line
        elif line.strip().endswith('.m4s'):
            if current_extinf:
                # Get the original watermarked segment this m4s came from
                original_segment = segment_mapping.get(line.strip(), "")
                segments_info.append({
                    'extinf': current_extinf,
                    'file': line.strip(),
                    'original_segment': original_segment
                })
                current_extinf = None
    
    # Create a mapping of selected segment files to their copy indexes
    selected_copy_indexes = {}
    if selected_segment_files:
        for file_path in selected_segment_files:
            basename = os.path.basename(file_path)
            copy_index = extract_copy_index(basename)
            segment_num = extract_segment_number(file_path)
            if segment_num is not None and copy_index is not None:
                selected_copy_indexes[segment_num] = copy_index
                logger.debug(f"Selected segment {segment_num} with copy {copy_index}")
    
    # Create custom playlist with just the segments we want
    custom_playlist_name = f"custom_{pattern_str}.m3u8"
    custom_playlist_path = os.path.join(hls_dir, custom_playlist_name)
    
    with open(custom_playlist_path, 'w') as f:
        # Write header
        for line in header_lines:
            f.write(f"{line}\n")
        
        # Write only the segments that match our selected segments
        segment_count = 0
        used_segments = set()
        
        for segment_info in segments_info:
            original_segment = segment_info['original_segment']
            m4s_file = segment_info['file']
            
            # Check if this m4s segment corresponds to one of our selected watermarked segments
            if original_segment in [os.path.basename(f) for f in selected_segment_files]:
                f.write(f"{segment_info['extinf']}\n")
                f.write(f"{m4s_file}\n")
                used_segments.add(original_segment)
                segment_count += 1
                logger.debug(f"Added {m4s_file} from {original_segment} to playlist")
        
        # Log segments that weren't found
        if selected_segment_files:
            selected_basenames = [os.path.basename(f) for f in selected_segment_files]
            unused_segments = set(selected_basenames) - used_segments
            if unused_segments:
                logger.warning(f"Could not find m4s segments for the following watermarked segments: {unused_segments}")
        
        # Write end tag
        f.write("#EXT-X-ENDLIST\n")
    
    logger.info(f"Created custom playlist with {segment_count} segments")
    
    # Update the master playlist to include the custom playlist
    custom_master_name = f"master_{pattern_str}.m3u8"
    custom_master_path = os.path.join(hls_dir, custom_master_name)
    
    # Copy and modify the master playlist
    with open(master_playlist_path, 'r') as src:
        master_content = src.read()
    
    # Update playlist reference in the master playlist
    updated_master = master_content.replace('playlist.m3u8', custom_playlist_name)
    
    with open(custom_master_path, 'w') as dst:
        dst.write(updated_master)
    
    # Create a simplified HTTP server configuration file to avoid CORS issues
    server_config_path = os.path.join(hls_dir, "cors_server.py")
    with open(server_config_path, 'w') as f:
        f.write("""#!/usr/bin/env python3
import http.server
import socketserver
import os

PORT = 8000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

handler = CORSHTTPRequestHandler

print(f"Starting HTTP server on port {PORT} with CORS support")
print(f"Current directory: {os.getcwd()}")
print(f"To view HLS stream, open URL: http://localhost:{PORT}/index.html")
print("Use Ctrl+C to stop the server")

with socketserver.TCPServer(("", PORT), handler) as httpd:
    httpd.serve_forever()
""")
    
    # Create a sample index.html with hls.js for easy playback
    index_path = os.path.join(hls_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>HLS Player - Watermarked with Pattern {pattern_str}</title>
    <style>
        body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
        #video {{ width: 100%; max-width: 800px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .info {{ margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 4px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
</head>
<body>
    <div class="container">
        <h1>HLS Player - Watermarked Stream</h1>
        <video id="video" controls></video>
        <div class="info">
            <p><strong>Fingerprint Pattern:</strong> {pattern_str}</p>
            <p><strong>Master Playlist:</strong> <a href="{custom_master_name}">{custom_master_name}</a></p>
            <p><strong>Custom Playlist:</strong> <a href="{custom_playlist_name}">{custom_playlist_name}</a></p>
            <p><strong>Instructions:</strong> This video has a unique watermark pattern. If shared, the source can be traced.</p>
        </div>
    </div>
    <script>
      document.addEventListener('DOMContentLoaded', function () {{
        var video = document.getElementById('video');
        var videoSrc = '{custom_master_name}';
        
        if (Hls.isSupported()) {{
          var hls = new Hls();
          hls.loadSource(videoSrc);
          hls.attachMedia(video);
          hls.on(Hls.Events.MANIFEST_PARSED, function() {{
            video.play();
          }});
        }}
        else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
          video.src = videoSrc;
          video.addEventListener('loadedmetadata', function() {{
            video.play();
          }});
        }}
        else {{
          console.error('HLS is not supported by your browser');
          document.write('<p style="color: red">Error: HLS is not supported by your browser</p>');
        }}
      }});
    </script>
</body>
</html>
""")
    
    # Make the server script executable
    os.chmod(server_config_path, 0o755)
    
    logger.info(f"Created custom HLS playlist: {custom_playlist_path}")
    logger.info(f"Created custom master playlist: {custom_master_path}")
    logger.info(f"Created HLS player: {index_path}")
    logger.info(f"To serve with CORS support, run: python {server_config_path}")
    
    return custom_master_path

def save_leak_info(output_dir, pattern, selected_files, custom_hls_path=None):
    """
    Save information about the generated leak
    
    Args:
        output_dir: Output directory
        pattern: Copy pattern used
        selected_files: List of selected segment files
        custom_hls_path: Path to custom HLS playlist if created
    """
    pattern_str = ''.join(map(str, pattern))
    
    info = {
        "copy_pattern": pattern,
        "pattern_string": pattern_str,
        "selected_segments": [os.path.basename(f) for f in selected_files]
    }
    
    if custom_hls_path:
        hls_dir = os.path.dirname(custom_hls_path)
        info["custom_hls_playlist"] = os.path.basename(custom_hls_path)
        info["playback_instructions"] = {
            "step1": "Start the CORS-enabled HTTP server",
            "command": f"cd {hls_dir} && python cors_server.py",
            "step2": "Open the following URL in your browser",
            "url": f"http://localhost:8000/index.html",
            "step3": "The video will play with your specific watermark pattern"
        }
    
    info_file = os.path.join(output_dir, "leak_info.json")
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Leak information saved to: {os.path.abspath(info_file)}")
    
    return info_file

def main():
    parser = argparse.ArgumentParser(description="Generate a leaked MP4 by selecting specific copies of each segment")
    parser.add_argument("copies_file", help="Path to segment_copies.json file")
    parser.add_argument("--output-file", help="Path to output leaked MP4 file (default: <copies_file_dir>/leaked_video.mp4)")
    parser.add_argument("--pattern", help="Pattern of copy indexes to use (e.g. '0123')")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducible random patterns")
    parser.add_argument("--create-hls", action="store_true", help="Create custom HLS playlists for browser playback")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server for HLS playback after creating files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--detect", action="store_true", help="Run detection on the generated leaked video")
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Get base output directory from copies_file path
    base_output_dir = os.path.dirname(os.path.abspath(args.copies_file))
    
    # Determine output file path
    if args.output_file:
        # If output file path is provided, ensure it's within the base output directory
        if os.path.isabs(args.output_file):
            # If absolute path, place it in the base output directory with the same filename
            output_file = os.path.join(base_output_dir, os.path.basename(args.output_file))
        else:
            # If relative path, join with base output directory
            output_file = os.path.join(base_output_dir, args.output_file)
    else:
        # Default to leaked_video.mp4 in the base output directory
        output_file = os.path.join(base_output_dir, "leaked_video.mp4")
    
    # Create detection directory inside the base output directory
    detection_dir = os.path.join(base_output_dir, "detection")
    os.makedirs(detection_dir, exist_ok=True)
    
    # Create segments directory inside detection directory
    detection_segments_dir = os.path.join(detection_dir, "segments")
    os.makedirs(detection_segments_dir, exist_ok=True)
    
    # Load segment copies information
    segment_copies_info = load_segment_copies(args.copies_file)
    
    # Select copies based on pattern or randomly
    selected_files, copy_pattern = select_copies(
        segment_copies_info,
        args.copies_file,
        pattern=args.pattern,
        random_seed=args.random_seed
    )
    
    # Get the pattern string
    pattern_str = ''.join(map(str, copy_pattern))
    
    # Log detailed info about selected files
    logger.debug(f"Selected {len(selected_files)} segment files with pattern {pattern_str}:")
    for i, file_path in enumerate(selected_files):
        file_name = os.path.basename(file_path)
        segment_num = extract_segment_number(file_path)
        copy_index = extract_copy_index(file_name)
        logger.debug(f"  Segment {segment_num} (copy {copy_index}): {file_name}")
    
    # Concatenate selected segments
    concatenate_segments(selected_files, output_file)
    
    # Create custom HLS playlist
    custom_hls_path = None
    if args.create_hls:
        try:
            custom_hls_path = create_custom_hls_playlist(base_output_dir, pattern_str, selected_files)
        except Exception as e:
            logger.error(f"Failed to create HLS playlist: {e}")
    
    # Save leak information to the base output directory
    save_leak_info(base_output_dir, copy_pattern, selected_files, custom_hls_path)
    
    # Run detection if requested
    if args.detect:
        try:
            payload_file = os.path.join(base_output_dir, 'segment_payloads.json')
            copies_file = args.copies_file
            
            detect_cmd = [
                'python', 'tests/detect_watermarks.py',
                output_file,
                detection_dir,
                '--payload-file', payload_file,
                '--copies-file', copies_file
            ]
            
            logger.info(f"Running detection on generated leaked video: {' '.join(detect_cmd)}")
            subprocess.run(detect_cmd, check=True)
            logger.info(f"Detection completed. Results saved to: {os.path.join(detection_dir, 'detection_results.json')}")
        except Exception as e:
            logger.error(f"Failed to run detection: {e}")
    
    # Print copy pattern
    logger.info(f"Generated leaked video with copy pattern: {pattern_str}")
    print(f"\nLeak fingerprint: {pattern_str}")
    print(f"This fingerprint can be used to identify the source of the leak")
    print(f"Leaked video saved to: {output_file}")
    
    if custom_hls_path:
        print(f"\nCustom HLS playlist created: {custom_hls_path}")
        print(f"To play the HLS stream in a browser:")
        print(f"1. Start the CORS-enabled HTTP server:")
        hls_dir = os.path.dirname(custom_hls_path)
        print(f"   cd {hls_dir} && python cors_server.py")
        print(f"2. Open in your browser: http://localhost:8000/index.html")
    
    # Start HTTP server if requested
    if args.serve and custom_hls_path:
        hls_dir = os.path.dirname(custom_hls_path)
        
        # Change directory to HLS directory
        current_dir = os.getcwd()
        abs_hls_dir = os.path.abspath(hls_dir)
        
        try:
            # Ensure the HLS directory exists
            if not os.path.exists(abs_hls_dir):
                print(f"Error: HLS directory '{abs_hls_dir}' not found")
                return
                
            # Change to the HLS directory
            os.chdir(abs_hls_dir)
            print(f"Changed directory to: {os.getcwd()}")
            
            # Verify the server script exists
            server_script = "cors_server.py"
            if not os.path.exists(server_script):
                print(f"Error: Server script '{server_script}' not found in {os.getcwd()}")
                return
            
            # Make the script executable
            os.chmod(server_script, 0o755)
            
            # Execute the server script
            print("\nStarting HTTP server for HLS playback...")
            subprocess.run(["python", server_script], check=True)
        except KeyboardInterrupt:
            print("\nServer stopped by user.")
        except Exception as e:
            print(f"Error starting HTTP server: {str(e)}")
            print(f"You can manually start the server with: cd {abs_hls_dir} && python cors_server.py")
        finally:
            # Always change back to the original directory
            os.chdir(current_dir)

if __name__ == "__main__":
    main() 