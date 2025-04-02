python tests/mark_video_to_hls.py tests/media/in.mp4 output
ffmpeg -i output/hls/playlist.m3u8 -c copy output/leaked_video.mp4
python tests/detect_watermarks.py output/leaked_video.mp4 output


### generate watermarked videos
python tests/mark_video_to_hls.py tests/media/in.mp4 output --copies 3 --clean
### Simulate a video leak with unique pattern
python tests/generate_leak.py output/segment_copies.json --pattern "0120" --create-hls --verbose
### detect the pattern
python tests/detect_watermarks.py output/leaked_video.mp4 detection --payload-file output/segment_payloads.json --copies-file output/segment_copies.json