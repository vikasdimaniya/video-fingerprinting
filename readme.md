
### Create hls stream
python hls_generator/video_to_hls.py
### Mark the video with forensic watermark
python mark/mark_m4s.py
### Start the server to serve hls stream
cd marked_output && python -m http.server 8000