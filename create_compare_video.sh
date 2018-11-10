python3 video_stabilization_algorithm.py -i data/hippo.mp4 -o data/stabilized.mp4
ffmpeg -i data/hippo.mp4 -i data/stabilized.mp4 -filter_complex hstack -y data/compare.mp4