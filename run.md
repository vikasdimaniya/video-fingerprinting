python tests/mark_video_to_hls.py tests/media/in.mp4 output
ffmpeg -i output/hls/playlist.m3u8 -c copy output/leaked_video.mp4
python tests/detect_watermarks.py output/leaked_video.mp4 output