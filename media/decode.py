import numpy as np
import struct
from pathlib import Path
import os

from common.base import offscreen_debug_log

class DecodeError(Exception):
    pass

class VideoData:
    width: int
    height: int
    frame_count: int
    timestamps: list[float]
    frame_index: int
    name: str
    audio_data: bytes

    def __init__(self, name, width, height, frame_count):
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.total_pixels = self.width * self.height
        self.frame_index = 0
        self.name = name
        self.timestamps = []
        self.audio_data = None

    def next_frame(self):
        return None

class ChotomateVideoData(VideoData):
    def __init__(self, path: str):
        try:
            with open(path, 'r') as file:
                data = file.readlines()

            width, height, frame_count = [int(x) for x in data[0].split(',')]
            super().__init__(Path(path).stem, width, height, frame_count)
            self.timestamps = [float(x) for x in data[1].split(',')]
            self.current_value = bool(int(data[2]))
            self.value_deltas = [int(x) for x in data[3].split(',')]
        except:
            raise DecodeError('Video file not recognized or corrupted: failed to parse the file')

        self.delta_index = 0
        self.frame_buffer = []
        self.reshape_order = 'C'

        audio_path = Path(path).parent / f'{Path(path).stem}.ogg'
        if os.path.exists(audio_path):
            print(f'Loading audio from {audio_path}...')
            with open(audio_path, 'rb') as file:
                self.audio_data = file.read()
        else:
            offscreen_debug_log('=' * 80 + f'\nError: could not find audio file at {audio_path}. Loading with no audio\n' + '=' * 80, also_print_now=True)

    # decode time: 7.8 seconds -> 5.7 seconds (5.1 with better downscaler)
    def next_frame(self):
        frame = np.zeros((self.total_pixels), dtype=np.bool_)
        frame_ptr = 0
        draw_mod = int(1 - self.current_value)

        while frame_ptr < self.total_pixels:
            delta = self.value_deltas[self.delta_index]
            if draw_mod == self.delta_index % 2:
                frame[frame_ptr:frame_ptr + delta] = 1
            frame_ptr += delta
            self.delta_index += 1

        residual_delta = delta - (frame_ptr - self.total_pixels)
        self.delta_index -= 1
        self.value_deltas[self.delta_index] -= residual_delta

        self.frame_index += 1
        return frame.reshape((self.height, self.width), order=self.reshape_order)

class Chotomate2VideoData(ChotomateVideoData):
    def __init__(self, path: str):
        with open(path, 'rb') as file:
            data = file.read()

        magic = b"\x89CHOTOMATE\x00BCVEFMT\xff"
        if data[:len(magic)] != magic:
            raise DecodeError('Video file not recognized: no magic')

        data_offset = len(magic)
        header_format = '<BHHIBIBIB'
        timestamp_format = '<B'
        reserved, width, height, frame_count, initial_value, delta_count, timestamp_resolution_us, audio_len, order = struct.unpack_from(header_format, data, data_offset)
        if reserved != 0:
            raise DecodeError(f'File version not supported: unexpected reserved value {reserved}')

        VideoData.__init__(self, Path(path).stem, width, height, frame_count)

        self.reshape_order = chr(order)
        self.current_value = bool(initial_value)
        data_offset += struct.calcsize(header_format)
        if audio_len > 0:
            self.audio_data = data[data_offset:data_offset + audio_len]
            data_offset += audio_len
            offscreen_debug_log(f'Read {audio_len/1024:0.1f} KB of audio stream', also_print_now=True)
        else:
            offscreen_debug_log('Warning: this file contains no audio', also_print_now=True)
        q_timestamps = [struct.unpack_from(timestamp_format, data, data_offset + i)[0] for i in range(frame_count - 1)]
        data_offset += len(q_timestamps) * struct.calcsize(timestamp_format)
        self.timestamps = list(np.cumsum([0] + q_timestamps) * timestamp_resolution_us / 1000 / 1000)
        self.value_deltas = []
        self.delta_index = 0
        self.frame_buffer = []
        for _ in range(delta_count):
            value, = struct.unpack_from('<B', data, data_offset)
            if value < 254:
                self.value_deltas.append(value + 1)
                data_offset += 1
            elif value == 254:
                value, = struct.unpack_from('<H', data, data_offset + 1)
                self.value_deltas.append(value + 255)
                data_offset += 3
            elif value == 255:
                value, = struct.unpack_from('<I', data, data_offset + 1)
                self.value_deltas.append(value)
                data_offset += 5
            else:
                raise RuntimeError('unreachable')

class AutoVideoData(VideoData):
    def __init__(self, path: str):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

        print(f'Loading video data from {path}')
        self.video_data = None
        try:
            self.video_data = Chotomate2VideoData(path)
        except DecodeError:
            pass

        try:
            self.video_data = ChotomateVideoData(path)
        except DecodeError:
            pass
        
        if self.video_data is None:
            raise DecodeError('Could not read the video file. The file is unsupported or corrupted')

    def next_frame(self):
        return self.video_data.next_frame()
    
    def __getattr__(self, name):
        whitelist = ['name', 'width', 'height', 'frame_index', 'frame_count', 'audio_data', 'timestamps']
        if name not in whitelist: raise AttributeError(name)
        return getattr(self.video_data, name)
