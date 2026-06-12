from common.base import *

from math import sqrt

class SpeedScene(CharBufferScene):
    def __init__(self, width: int, height: int, char_pallet: CharPallet, args: argparse.Namespace):
        super().__init__(width, height, char_pallet)
        self.char_buffer.set_color_buffer_evolver(0, ColorEvolvers.power_fade(fade_power=args.fade_power))
        self.primary_color = args.primary_color
        self.spawn_limit = args.spawn_limit
        self.particle_limit = args.particle_limit
        self.move_interval_factor = args.interval_factor
        self.min_spawn_radius = height * args.min_spawn_radius
        self.max_spawn_radius = height * args.max_spawn_radius

    def register_arg_parser(subparser: argparse.ArgumentParser):
        subparser.add_argument('--fade-power', type=custom_float(min_value=1, min_inclusive=False), default=1.5, help='|')
        subparser.add_argument('--spawn-limit', type=positive_int, default=5, help='|')
        subparser.add_argument('--particle-limit', type=positive_int, default=500, help='|')
        subparser.add_argument('--interval-factor', type=positive_float, default=0.5, help='|')
        subparser.add_argument('--min-spawn-radius', type=float01, default=0.12, help='|')
        subparser.add_argument('--max-spawn-radius', type=float01, default=0.65, help='|')
        subparser.add_argument('--primary-color', type=color_arg, default=color8(128, 200, 255), help='6 character color code, e.g. ffffff')

    def get_direction(self, x, y):
        dir_x, dir_y = self.width / 2 - x, self.height / 2 - y
        magnitude = sqrt(dir_x * dir_x / 4 + dir_y * dir_y)
        return dir_x / magnitude, dir_y / magnitude, magnitude

    def evolve_state(self, char_buffer: CharBuffer, char_state: CharState, x: int, y: int, tag: int, fx: float, fy: float, move_interval: float):
        dir_x, dir_y, magnitude = self.get_direction(fx, fy)
        velocity = pow(dir_x / 2, 2) + dir_y * dir_y
        h_factor = 1 / velocity

        fx -= h_factor * dir_x
        fy -= h_factor * dir_y
        char_state.data = (fx, fy, move_interval)

        char_state.x = round(fx)
        char_state.y = round(fy)

        if char_state.x < 0 or char_state.y < 0 or char_state.x >= char_buffer.width or char_state.y >= char_buffer.height:
            return False, 0 # destroy if out of bounds

        char_buffer.update_char_position(char_state, '*')

        return True, perf_counter() + move_interval

    @return_execution_time()
    def spawn_chars(self, char_buffer: CharBuffer):
        max_spawn_count = min(self.spawn_limit, self.particle_limit - len(char_buffer.char_states))

        for _ in range(random.randrange(0, max_spawn_count + 1)):
            start_x, start_y = random.randrange(0, self.width), random.randrange(0, self.height)
            _, _, distance = self.get_direction(start_x, start_y)
            if distance < self.min_spawn_radius or distance > self.max_spawn_radius:
                continue

            move_interval = random.uniform(0.015, 0.02) * self.move_interval_factor
            char_state = CharState(start_x, start_y, self.primary_color, 0, self.evolve_state, (start_x, start_y, move_interval,))
            char_buffer.add_char_state(char_state, update_position=False)
            char_buffer.update_char_position(char_state, '*')

    def do_frame(self, terminal: Terminal, dt: float):
        spawn_time = self.spawn_chars(self.char_buffer)
        render_time = self.char_buffer.render_chars(terminal)
        color_time = self.char_buffer.evolve_color(dt)
        evolve_time = self.char_buffer.evolve_state()

        return [spawn_time, render_time, color_time, evolve_time], f' ({len(self.char_buffer.char_states)}) [Speed]'

def get_scene_list():
    return {'speed': SpeedScene}
