import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from media.common import decode_video, quantize_video, print_progress
from media.encode import compress_color_stripes_continuous

# ================================================================

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <path>')
    sys.exit(2)

path = sys.argv[1]
frames, timestamps = decode_video(path)
binary_frames = quantize_video(frames)

print('--------- Decoding done, exporting...  ----------')
output_file = 'bad-apple.chotomate'

with open(output_file, 'w') as output:
    height, width = binary_frames[0].shape
    total_pixels = height * width
    output.write(f'{width},{height},{len(binary_frames)}\n')
    output.write(','.join(f'{x/1000:0.3f}' for x in timestamps) + '\n')
    current_value = binary_frames[0][0][0]
    output.write(f'{int(current_value)}\n')
    print(f'Header done... (IC={int(current_value)})')

    current_count = 0
    deltas = []
    for i, frame in enumerate(binary_frames):
        print_progress(i, len(binary_frames), 'Encoding frame')
        ds, current_value, current_count = compress_color_stripes_continuous(frame.ravel(), current_value, current_count, total_pixels)
        deltas += ds

    if current_count > 0:
        deltas.append(current_count)

    output.write(','.join(str(x) for x in deltas))
    output.write('\n')

print(f'Done. Wrote {output_file} (deltas: {len(deltas)})')
print(f'Validation: {height * width * len(binary_frames)} == {np.sum(deltas)}? {height * width * len(binary_frames) == np.sum(deltas)}')
