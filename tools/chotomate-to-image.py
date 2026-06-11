import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from media.decode import AutoVideoData

input_path = 'videos/bad-apple.chotomate2'

video = AutoVideoData(input_path)

plt.ion()

fig, ax = plt.subplots()
img = ax.imshow(video.next_frame())

# PSA: this might or might not be problematic to kill
while video.frame_index < video.frame_count:
    print(f'Frame {video.frame_index}...')
    frame = video.next_frame()
    img.set_data(frame)
    img.autoscale()
    fig.canvas.draw_idle()
    plt.pause(0.01)

