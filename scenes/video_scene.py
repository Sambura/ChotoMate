from common.base import *
from media.decode import VideoData, AutoVideoData
from media.common import print_progress, downscale_horrendous, downscale_pil

from pathlib import Path
import os
import io

import pygame

def load_frames_from_cache(video_data: VideoData, cache_path: str):
    frames = []

    if os.path.exists(cache_path):
        print(f'Restoring frames from cache...', end='\r')
        frame_cache = np.load(cache_path)
        frame_count, height, width = frame_cache.shape
        if frame_count != video_data.frame_count:
            offscreen_debug_log(f'Invalid cache at {cache_path}: frame count mismatch. Removing...', also_print_now=True)
            return []

        for i in range(frame_count):
            frames.append((frame_cache[i], video_data.timestamps[i]))
            print_progress(i, frame_count, 'Restoring frames from cache', f'Resolution: {width}x{height}')

    return frames

@return_execution_time(has_return=True)
def generate_frames(video_data: VideoData, new_width: int, new_height: int, char_width: int, create_cache: bool, downscaler=None):
    if downscaler is None: downscaler = downscale_pil
    cache_name = f'.{video_data.name}-{new_width}x{new_height}-{char_width}.chotomate.cache.npy'
    frames = load_frames_from_cache(video_data, cache_name)
    if len(frames) == video_data.frame_count:
        return frames

    original_aspect = video_data.width / video_data.height
    new_aspect = new_width / new_height
    aspect_width, aspect_height = new_width, new_height
    if new_aspect > original_aspect:
        # double the width if we use narrow characters
        aspect_width = (2 // char_width) * round(new_height * original_aspect)
        offscreen_debug_log(f'Calculated aspect_width: {new_width} -> {aspect_width} (accounted for {char_width}-wide characters)', also_print_now=True)
    elif new_aspect < original_aspect:
        aspect_height = round(new_width / original_aspect)

    aspect_zero_x = (new_width - aspect_width) // 2
    aspect_zero_y = (new_height - aspect_height) // 2
    cache_msg = '' if create_cache else '. Use --cache-frames to speed this up next time'

    while video_data.frame_index < video_data.frame_count:
        timestamp = video_data.timestamps[video_data.frame_index]
        frame_full = video_data.next_frame()

        print_progress(video_data.frame_index - 1, video_data.frame_count, 'Generating frames', f'Resolution: {new_width}x{new_height}{cache_msg}')

        frame_holder = np.zeros((new_height, new_width), dtype=np.bool_)
        frame = downscaler(frame_full, aspect_width, aspect_height)
        frame_holder[aspect_zero_y:aspect_zero_y+aspect_height, aspect_zero_x:aspect_zero_x+aspect_width] = frame

        frames.append((frame_holder, timestamp))

    if create_cache:
        np.save(cache_name, np.array([x for x, y in frames]))

    return frames

def draw_no_source(char_buffer: CharBuffer, runtime: float):
    char_buffer.clear_char_buffer()

    blank_frame = (runtime / 1.4) % 1 > 0.55
    if blank_frame: return

    x_center, y_center = char_buffer.width // 2, char_buffer.height // 2
    color = color8(255, 20, 20)
    no_source_text = '[NO SOURCE]'
    time_text = f'{runtime:0.2f}'
    line_text = '=' * (char_buffer.width // 3)
    border_text = f'={" " * (len(line_text) - 2)}='

    # border
    char_buffer.print_line_to_char_buffer(line_text, x_center, y_center - 3, color, centered=True)
    for y in range(y_center - 2, y_center + 3):
         char_buffer.print_line_to_char_buffer(border_text, x_center, y, color, centered=True)
    char_buffer.print_line_to_char_buffer(line_text, x_center, y_center + 3, color, centered=True)

    # texts
    char_buffer.print_line_to_char_buffer(no_source_text, x_center, y_center - 1, color, centered=True)
    char_buffer.print_line_to_char_buffer(time_text, x_center, y_center + 1, color, centered=True)

class BaseVideoScene(CharBufferScene):
    def __init__(self, width: int, height: int, char_pallet: CharPallet, args: argparse.Namespace, primary_color: np.ndarray):
        super().__init__(width, height, char_pallet)
        self.video_name = Path(args.filename).stem
        self.primary_color = primary_color

        video = AutoVideoData(args.filename)
        downscaler = downscale_horrendous if args.fast_downscale else downscale_pil
        self.frames, elapsed = generate_frames(video, self.width, self.height, self.char_buffer.char_pallet.char_width, create_cache=args.cache_frames, downscaler=downscaler)
        print(f'Time elapsed: {elapsed:0.1f}s')
        self.next_frame = 0
        self.start_time = perf_counter() + 1.5
        self.current_frame = self.frames[0][0]

        # set buffer to 128 seems to remove audio delay? not sure
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=128)
        pygame.mixer.init()
        if video.audio_data is not None:
            self.audio = pygame.mixer.Sound(io.BytesIO(video.audio_data))
            self.audio.set_volume(args.audio_volume)
        else:
            self.audio = pygame.mixer.Sound(np.zeros((44100), dtype=np.float16))

    def register_arg_parser(subparser: argparse.ArgumentParser):
        subparser.add_argument('filename', type=str)
        subparser.add_argument('--audio-volume', '-v', type=float01, default=0.5, help='|')
        subparser.add_argument('--cache-frames', action='store_true', help='|')
        subparser.add_argument('--fast-downscale', action='store_true', help='|')

# --no-ghosting --fade 0.8 --max-fall 5 --max-spawn 10000
class VideoScene(BaseVideoScene):
    def __init__(self, width: int, height: int, char_pallet: CharPallet, args: argparse.Namespace):
        super().__init__(width, height, char_pallet, args, color8(255, 255, 255))

        self.saturation_density = args.saturation_density
        self.fade_rate = args.fade_rate
        self.max_fall_delay = args.max_fall_delay
        self.no_ghosting = args.no_ghosting
        self.max_spawn_per_frame = args.max_spawn_per_frame
        self.spawn_limit = args.spawn_limit
        self.frame_status = '(syncing)'

        self.char_buffer.set_color_buffer_evolver(0, lambda x, y: x, evolve_dead=False)
        self.char_buffer.set_color_buffer_evolver(0, self.color_fade, evolve_alive=False)

    def register_arg_parser(subparser: argparse.ArgumentParser):
        BaseVideoScene.register_arg_parser(subparser)
        subparser.add_argument('--saturation-density', '-d', type=float01, default=0.5, help='|')
        subparser.add_argument('--fade-rate', '-r', type=positive_float, default=1.5, help='|')
        subparser.add_argument('--max-fall-delay', type=positive_int, default=3, help='|')
        subparser.add_argument('--no-ghosting', action='store_true', help='|')
        subparser.add_argument('--max-spawn-per-frame', type=positive_int, default=1000, help='|')
        subparser.add_argument('--spawn-limit', type=positive_int, default=25000, help='|')

    def color_fade(self, color_buffer: np.ndarray[float], delta_time: float):
        return np.maximum(0, color_buffer - np.power(color_buffer, 1.3) * self.fade_rate * delta_time)

    def evolve_state(self, char_buffer: CharBuffer, char_state: CharState, x: int, y: int, tag: int):
        if y >= self.height - 1:
            return False, 0 # below screen, destroy

        if not self.current_frame[y][x]: # or random.random() > 0.95:
            # character destroyed by dark pixel: set tag to -100 to immediately set to black on next redraw
            self.char_buffer.tag_buffer[char_state.y, char_state.x] = -100
            return False, 0

        char_state.y += 1
        self.char_buffer.update_char_position(char_state, self.char_buffer.get_random_char())

        next_frame = min(len(self.frames) - 1, self.next_frame + random.randrange(0, self.max_fall_delay))
        return True, self.frames[next_frame][1] + self.start_time

    @return_execution_time()
    def spawn(self, char_buffer: CharBuffer):
        free_slots_y, free_slots_x = np.nonzero(char_buffer.tag_buffer < 0)
        frame_data = self.current_frame[free_slots_y, free_slots_x]

        free_slots = [(x, y) for f, x, y in zip(frame_data, free_slots_x, free_slots_y) if f]

        occupied_slots = self.width * self.height - len(free_slots_x)
        total_frame_slots = occupied_slots + len(free_slots)
        max_occupied_slots = int(total_frame_slots * self.saturation_density)
        slots_remaining = max(0, max_occupied_slots - occupied_slots)

        spawn_count = max(min(self.spawn_limit - len(self.char_buffer.char_states), self.max_spawn_per_frame), 0)
        spawn_count = min(spawn_count, len(free_slots), slots_remaining)

        spawn_points = random.sample(free_slots, spawn_count)

        for start_x, start_y in spawn_points:
            char_buffer.add_char_state(CharState(start_x, start_y, self.primary_color, 0, self.evolve_state))

    @return_execution_time()
    def anti_ghosting(self):
        dark_pixels_mask = self.current_frame == 0
        white_pixels_mask = self.current_frame != 0
        non_existing_tag = -100 # tag with no color evolver

        self.char_buffer.tag_buffer = white_pixels_mask * self.char_buffer.tag_buffer + non_existing_tag * dark_pixels_mask

    def do_frame(self, terminal: Terminal, delta_time: float):
        anti_ghosting_time = 0
        while perf_counter() >= self.frames[self.next_frame][1] + self.start_time:
            if self.next_frame == 0:
                self.audio.play()

            if self.next_frame + 1 >= len(self.frames):
                draw_no_source(self.char_buffer, perf_counter() - self.start_time - self.frames[-1][1])
                self.char_buffer.render_chars(terminal)
                return [], ' [no source]'

            self.current_frame = self.frames[self.next_frame][0]
            self.next_frame += 1
            self.frame_status = f'{self.next_frame}/{len(self.frames)}'

            if self.no_ghosting:
                anti_ghosting_time = self.anti_ghosting()

        spawn_time = self.spawn(self.char_buffer)
        render_time = self.char_buffer.render_chars(terminal)
        color_time = self.char_buffer.evolve_color(delta_time) + anti_ghosting_time
        evolve_time = self.char_buffer.evolve_state()

        return [spawn_time, render_time, color_time, evolve_time], f' [{self.video_name}]: {len(self.char_buffer.char_states)} chr, {self.frame_status}'

class PlainVideoScene(BaseVideoScene):
    def __init__(self, width: int, height: int, char_pallet: CharPallet, args: argparse.Namespace):
        super().__init__(width, height, None, args, color8(255, 255, 255)) # ignore pallet

        self.white_frame = np.ones((height, width, 3), dtype=float) * 255
        self.display_frame()

    @return_execution_time()
    def display_frame(self):
        self.char_buffer.color_buffer = self.white_frame * self.current_frame[:, :, np.newaxis]

    def do_frame(self, terminal: Terminal, dt: float):
        new_frame = False
        while perf_counter() >= self.frames[self.next_frame][1] + self.start_time:
            new_frame = True
            if self.next_frame == 0:
                self.audio.play()

            if self.next_frame + 1 >= len(self.frames):
                draw_no_source(self.char_buffer, perf_counter() - self.start_time - self.frames[-1][1])
                self.char_buffer.render_chars(terminal)
                return [], ' [no source]'

            self.current_frame = self.frames[self.next_frame][0]
            self.next_frame += 1

        display_time = 0
        if new_frame:
            display_time = self.display_frame()
        render_time = self.char_buffer.render_color_buffer(terminal)

        return [display_time, render_time], f' [{self.video_name}]: '

def get_scene_list():
    return {
        'video': VideoScene,
        'flat-video': PlainVideoScene
    }
