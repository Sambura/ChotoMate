from common.base import *
from pathlib import Path
import importlib
import sys

def scan_scenes(scene_list: dict[str, Scene]):
    "Find scenes under `scenes/` and add them to `scene_list`"
    scenes_path = Path(__file__).resolve().parent / 'scenes'
    if not scenes_path.is_dir():
        print_error(f'Scene loading error: could not find directory `{scenes_path}`')
        return

    loaded_count = 0
    for name in scenes_path.iterdir():
        if not name.is_file() or name.suffix != '.py' or name.name == '__init__.py':
            continue

        try:
            module = importlib.import_module(f'{scenes_path.name}.{name.stem}')

            func_name = 'get_scene_list'
            get_scenes = getattr(module, func_name, None)
            if get_scenes is None:
                print_error(f'Error: Failed to load scenes from {name}: {func_name} not found')
                continue

            scenes = get_scenes()
            for scene_name, scene in scenes.items():
                if scene_name in scene_list:
                    print_error(f'Error: attempted to load scene {scene_name} from {name}, but the scene with such name already exists')
                    continue

                scene_list[scene_name] = scene
                loaded_count += 1

        except ImportError as e:
            print_error(f'Error: Failed to load scenes from {name}: {e}')
            continue

    print(f'scan_scenes: {loaded_count} scenes loaded')

# ===========================================================
# Classic scene
# ===========================================================

class ClassicScene(CharBufferScene):
    _spawners = { 'balanced': lambda self: self.spawn_balanced, 'top': lambda self: self.spawn_top }

    def __init__(self, width: int, height: int, char_pallet: CharPallet, args: argparse.Namespace):
        super().__init__(width, height, char_pallet)

        self.red_chance_threshold = 0.999999999
        self.red_status = 0
        self.char_buffer.set_color_buffer_evolver(0, ColorEvolvers.exponential_fade(fade_rate=args.fade_rate))
        self.spawn_func = ClassicScene._spawners[args.spawn_method](self)
        self.max_spawn = args.max_spawn

    def register_arg_parser(subparser: argparse.ArgumentParser):
        subparser.add_argument('--fade-rate', type=float01, default=0.95, help='|')
        subparser.add_argument('--spawn-method', type=str, default='balanced', choices=list(ClassicScene._spawners.keys()), help='|')
        subparser.add_argument('--max-spawn', type=positive_int, default=10, help='|')

    def evolve_state(char_buffer: CharBuffer, char_state: CharState, x: int, y: int, tag: int, fall_interval: float):
        if y >= char_buffer.height - 1:
            return False, 0 # below screen, destroy

        if random.random() > 0.99:
            return False, 0 # random disappearance

        char_state.y += 1
        char_buffer.update_char_position(char_state, char_buffer.get_random_char())

        return True, perf_counter() + fall_interval

    def spawn_character(self, x, y):
        do_green = random.random() < self.red_chance_threshold
        off_green = random.randrange(0, 130)
        color = color8(off_green, 255, off_green) if do_green else color8(255, 0, 116) # color8(255, 50, 50)
        fall_interval = random.uniform(0.015, 0.1)

        self.char_buffer.add_char_state(CharState(x, y, color, 0, ClassicScene.evolve_state, (fall_interval,)))

    def spawn_balanced(self):
        top_bias = self.height * 6 // 10   # bias to spawning chars at the top
        max_y = self.height * 9 // 10      # do not spawn chars too low
        start_x, start_y = random.randrange(0, self.width), random.randrange(-top_bias, max_y)
        if start_y < 0: start_y = 0

        self.spawn_character(start_x, start_y)

    def spawn_top(self):
        start_x, start_y = random.randrange(0, self.width), 1
        self.spawn_character(start_x, start_y)

    @return_execution_time()
    def spawn(self):
        max_spawn = int(self.max_spawn / 2 + self.max_spawn * self.red_chance_threshold / 2)

        for _ in range(random.randrange(0, max_spawn)):
            self.spawn_func()

    def advance_red_threshold(self):
        if self.red_status == 0:
            self.red_chance_threshold = pow(self.red_chance_threshold, 1.007)
            if self.red_chance_threshold < 0.000000001:
                self.red_status = 1
        elif self.red_status == 1:
            self.red_chance_threshold += 0.000001
            if self.red_chance_threshold > 0.0015:
                self.red_chance_threshold += 0.01
            if self.red_chance_threshold > 1:
                self.red_status = 0
                self.red_chance_threshold = 0.999999999
        else:
            self.red_status = 0

    def do_frame(self, terminal: Terminal, delta_time: float):
        self.advance_red_threshold()

        spawn_time = self.spawn()
        render_time = self.char_buffer.render_chars(terminal)
        evolve_time = self.char_buffer.evolve_state()
        color_time = self.char_buffer.evolve_color(delta_time)

        return [spawn_time, render_time, evolve_time, color_time], f' T: {self.red_chance_threshold:0.5f}'

# ===========================================================
# Main
# ===========================================================

def main_render_loop(terminal: Terminal, scene: Scene, target_fps: int, status_mode: str):
    sync_time = 0
    flush_time = 0
    actual_frame_time = 0
    frame_start = 0
    actual_fps = 0
    last_frame = perf_counter()
    full_status = status_mode == 'detailed'
    short_status = status_mode == 'fps'

    enable_timings_smoothing = True
    timings_smooth = None

    while True:
        # frame timings
        actual_frame_time = perf_counter() - frame_start
        frame_start = perf_counter()
        presentation_time = frame_start + 1 / target_fps - flush_time

        # compute fps
        actual_fps = smooth_value(actual_fps, 1 / actual_frame_time if actual_frame_time != 0 else 0)

        # render frame
        frame_time = perf_counter()
        timings, postfix = scene.do_frame(terminal, frame_time - last_frame)
        last_frame = frame_time

        if full_status:
            # smooth timings
            if enable_timings_smoothing:
                if timings_smooth is None:
                    timings_smooth = timings.copy()

                timings = timings_smooth = [smooth_value(x, y) for x, y in zip(timings_smooth, timings)]

            # Print status. Fps ans sync times are from the previous frame
            breakdown_str = '/'.join(f'{1000*x:2.0f}ms' for x in timings)
            terminal.print_status(f'Status: FPS {actual_fps:0.1f}/{target_fps:0.1f} | {terminal.width}x{terminal.height} | {breakdown_str}; sync: {1000*sync_time:2.1f}ms; flush: {1000*flush_time:0.1f}ms{postfix}')
        elif short_status:
            terminal.print_status(f'FPS {actual_fps:3.1f}')

        # wait until presentation time
        sync_time = smooth_value(sync_time, presentation_time - perf_counter())
        while presentation_time > perf_counter() + 0.00001:
            pass

        # present frame
        flip_duration = terminal.flip()
        flush_time = smooth_value(flush_time, flip_duration)

scenes = {
    'classic': ClassicScene,
    'test': TestScene
}

class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

if __name__ == '__main__':
    scan_scenes(scenes)

    desc_message = f'Wait a minute...\n\nUse `{sys.argv[0]} <scene_name> --help` for scene-specific help'
    parser = argparse.ArgumentParser(description=desc_message, formatter_class=HelpFormatter)
    parser.add_argument('--no-buffering', action='store_true', help='disable double buffering')
    parser.add_argument('--width', type=positive_int, help='set custom scene width. Defaults to current terminal width')
    parser.add_argument('--height', type=positive_int, help='set custom scene height. Defaults to current terminal height')
    parser.add_argument('--target-fps', type=int, default=60, help='set frame rate limit. Set to -1 for unlimited')
    parser.add_argument('--status', type=str, default='detailed', choices=['hide', 'fps', 'detailed'], help='status bar display mode')
    parser.add_argument('--pallet', type=str, choices=[x for x in char_pallets if x], help='change default character pallet. Ignored by some scenes')

    scene_subparsers = parser.add_subparsers(dest='scene_name', required=False)
    subparsers = { scene_name: scene_subparsers.add_parser(scene_name, formatter_class=HelpFormatter) for scene_name in scenes }

    for scene_name, subparser in subparsers.items():
        scenes[scene_name].register_arg_parser(subparser)

    # some insane hacks just so that you don't have to specify a scene to launch classic scene
    base_args, unknown = parser.parse_known_args()
    scene_args = base_args

    default_scene = 'classic'
    if base_args.scene_name is None:
        base_args.scene_name = default_scene
        scene_args = subparsers[default_scene].parse_args(unknown)
    elif len(unknown) > 0:
        raise RuntimeError(f'Unrecognized arguments: {unknown}. Make sure to specify global arguments before scene name')

    pallet = char_pallets[base_args.pallet]

    try:
        with Terminal(double_buffering=not base_args.no_buffering) as terminal:
            offscreen_debug_log(f'Loading scene {base_args.scene_name}...', also_print_now=True)
            width = terminal.width if base_args.width is None else base_args.width
            height = terminal.height if base_args.height is None else base_args.height

            try:
                scene = scenes[base_args.scene_name](width, height, pallet, args=scene_args)
            except Exception as e:
                offscreen_debug_log(f'Scene load failure: {e}')
                raise e

            main_render_loop(terminal, scene, target_fps=base_args.target_fps, status_mode=base_args.status)

    except KeyboardInterrupt:
        print('Interrupted (Ctrl+C)')
    except Exception as e:
        print('Unhandled exception occurred:\n\n' + '=' * 80 + '\nException details:\n')
        raise e
