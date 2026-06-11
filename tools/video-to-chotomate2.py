import sys
import numpy as np
import struct
from math import ceil
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from media.common import decode_video, quantize_video, print_progress
from media.encode import compress_color_stripes_continuous

def encode_timestamps(timestamps: list[float]) -> tuple[list[int], int]:
    timestamp_deltas_us = 1000 * (np.array(timestamps[1:]) - np.array(timestamps[:-1]))
    max_t_delta = np.max(timestamp_deltas_us)
    us_timestamp_resolution = ceil(max_t_delta / 255)
    assert us_timestamp_resolution < 256, 'FPS too low?'
    print(f'Max timestamp delta: {max_t_delta}us. Timestamp resolution: {us_timestamp_resolution}us')
    ts_deltas = []
    current_q_ts = 0
    for delta, abs_ts in zip(timestamp_deltas_us, timestamps[:-1]):
        q_delta = int(delta / us_timestamp_resolution) # undershoot deltas

        new_ts_us = (current_q_ts + q_delta) * us_timestamp_resolution
        error = 1000 * abs_ts - new_ts_us

        # increase delta if accumulated error too high
        while q_delta < 255 and error > us_timestamp_resolution / 2:
            q_delta += 1
            error -= us_timestamp_resolution

        ts_deltas.append(q_delta)
        current_q_ts += q_delta

        if error > 0.75 * us_timestamp_resolution:
            print(f'High frame timestamp error: {int(error)} us')

    return ts_deltas, us_timestamp_resolution

def encode_frames(frames, initial_value, order):
    total_pixels = frames[0].shape[0] * frames[0].shape[1]
    total_encoded_size = 0

    current_value = initial_value
    current_count = 0
    deltas = []
    for i, frame in enumerate(frames):
        ds, current_value, current_count = compress_color_stripes_continuous(frame.ravel(order=order), current_value, current_count, total_pixels)
        deltas += ds

        halfs = len([x for x in ds if 254 < x < 65536 + 255])
        ints = len([x for x in ds if x > 65536 + 255])
        frame_size = len(ds) + 2 * halfs + 4 * ints
        total_encoded_size += frame_size
        avg = total_encoded_size / (i + 1)

        print_progress(i, len(frames), 'Encoding frame', f'current frame: ~ {frame_size} bytes, avg: {avg:0.1f} bytes')

    if current_count > 0:
        deltas.append(current_count)

    return deltas

def process_video(path, audio_path):
    if audio_path is not None:
        with open(audio_path, 'rb') as file:
            audio_data = file.read()
    else:
        audio_data = None

    frames, timestamps = decode_video(path)
    frames = quantize_video(frames)

    # general info
    height, width = frames[0].shape
    initial_value = int(bool(frames[0][0, 0]))
    frames_count = len(frames)

    # compute time data
    ts_deltas, ts_resolution = encode_timestamps(timestamps)

    # compute frame data
    ravel_order = 'F'
    frame_deltas = encode_frames(frames, initial_value, order=ravel_order)

    # write out
    out_path = f'{str(Path(path).stem)}.chotomate2'
    magic = b"\x89CHOTOMATE\x00BCVEFMT\xff"
    
    with open(out_path, 'wb') as f:
        print('Writing header...')
        f.write(magic)
        reserved = 0
        audio_len = 0 if audio_data is None else len(audio_data)
        header = struct.pack('<B HH IBIB IB', reserved, width, height, frames_count, initial_value, len(frame_deltas), ts_resolution, audio_len, ord(ravel_order))
        f.write(header)

        print('Writing audio data...')
        if audio_data is not None:
            f.write(audio_data)

        print('Writing time data...')
        for ts_delta in ts_deltas:
            f.write(struct.pack('<B', ts_delta))

        print('Writing frame data...')
        for delta in frame_deltas:
            assert 0 < delta < 4294967296 # positive and can fit in uint32

            if delta < 255: # deltas: 0..254; byte value: 0..253
                packed_delta = struct.pack('<B', delta - 1)
            elif delta < 65536 + 255: # deltas: 255..65535 + 255
                packed_delta = struct.pack('<BH', 254, delta - 255)
            else:
                packed_delta = struct.pack('<BI', 255, delta)

            f.write(packed_delta)

    print(f'Wrote {out_path}: {width}x{height}, frames: {frames_count}')

if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <video_path> [audio_path]")
        sys.exit(2)

    audio_path = None if len(sys.argv) == 2 else sys.argv[2]
    process_video(sys.argv[1], audio_path)
