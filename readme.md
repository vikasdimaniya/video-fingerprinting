
### Create hls stream
python hls_generator/video_to_hls.py tests/media/in.mp4 --output-dir tests/hls_output

### Start the server to serve UN-MARKED hls stream 
cd tests/hls_output && python -m http.server 8000

### Mark the video with forensic watermark
python mark/hls_mark.py

### Start the server to server MARKED hls stream
cd out/marked_segments && python -m http.server 9000