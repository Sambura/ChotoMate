from common.base import *

class AsteroidScene(CharBufferScene):
    SURROUND_TAG = 0
    STATIONARY_TAG = 1
    FLOWING_TAG = 2

    def __init__(self, width: int, height: int, char_pallet: CharPallet, args: argparse.Namespace):
        super().__init__(width, height, char_pallet)
        self.char_buffer.set_color_buffer_evolver(AsteroidScene.SURROUND_TAG, ColorEvolvers.exponential_fade(fade_rate=0.98))
        self.char_buffer.set_color_buffer_evolver(AsteroidScene.STATIONARY_TAG, ColorEvolvers.exponential_fade(fade_rate=0.997))
        self.char_buffer.set_color_buffer_evolver(AsteroidScene.FLOWING_TAG, ColorEvolvers.exponential_fade(fade_rate=0.96))
        self.surround_color     = color8(211, 50, 151) # color8(253, 61, 181)
        self.stationary_color   = color8(28, 179, 236)
        self.flowing_color      = color8(255, 213, 21)
        self.max_spawn = 150
        self.fall_interval_factor = 0.8
        self.radius = 13

    def verify_bounds(self, x: int, y: int, tag: int) -> tuple[bool, bool]:
        if y >= self.height - 1:
            return False, False # below screen

        cx, cy = self.width // 2, self.height // 2
        in_circle = pow(cx - x, 2) / 4 + pow(cy - y, 2) < self.radius * self.radius
        in_band = abs(cx - x) < 2 * self.radius
        under_circle = in_band and y > cy

        if in_circle and tag == AsteroidScene.SURROUND_TAG:
            return False, False # remove primary color from the circle
        
        if not in_band and tag != AsteroidScene.SURROUND_TAG:
            return False, False # remove all non primary colors outside the band
        
        in_ellipse = pow(cx - x, 2) / 4 + pow(cy - y - 10, 2) * 2 < self.radius * 12
        if (not in_circle or not in_ellipse) and tag == 1:
            return False, False # remove liquid color from outside the circle or outside the ellipse
        
        flowing_ok = (in_circle and not in_ellipse) or under_circle
        return True, flowing_ok

    def evolve_state(self, char_buffer: CharBuffer, char_state: CharState, x: int, y: int, tag: int, fall_interval: float):
        if random.random() > 0.99:
            return False, 0 # random disappearance

        in_bounds, can_flow = self.verify_bounds(x, y, tag)
        if not in_bounds:
            return False, 0

        char_state.y += 1
        if tag == AsteroidScene.FLOWING_TAG and not can_flow: # hide (do not update) flowing color if it's not supposed to be there
            return True, perf_counter() + fall_interval

        char_buffer.update_char_position(char_state, char_buffer.get_random_char())

        if tag == AsteroidScene.STATIONARY_TAG:
            if random.random() < 0.1: # population control (TM)
                return False, 0

            return True, perf_counter() + random.uniform(2, 10)

        return True, perf_counter() + fall_interval

    @return_execution_time()
    def spawn_chars(self, char_buffer: CharBuffer):
        top_bias = self.height * 6 // 10   # bias to spawning chars at the top
        max_y = self.height * 9 // 10      # do not spawn chars too low

        for _ in range(random.randrange(0, self.max_spawn)):
            start_x, start_y = random.randrange(0, self.width), random.randrange(-top_bias, max_y)
            if start_y < -1: start_y = -1 # characters first render *after* they fall down one row, so we spawn them one row higher

            selector = random.random()
            if selector < 0.2:
                color, tag = self.surround_color, AsteroidScene.SURROUND_TAG
                fall_interval = random.uniform(0.01, 0.18) * self.fall_interval_factor
            elif selector < 0.85:
                color, tag = self.stationary_color, AsteroidScene.STATIONARY_TAG
                fall_interval = -1 # not applicable
            else:
                color, tag = self.flowing_color, AsteroidScene.FLOWING_TAG
                fall_interval = random.uniform(0.015, 0.1) * self.fall_interval_factor

            char_buffer.add_char_state(CharState(start_x, start_y, color, tag, self.evolve_state, (fall_interval,)), update_position=False)

    def do_frame(self, terminal: Terminal, dt: float):
        spawn_time = self.spawn_chars(self.char_buffer)
        render_time = self.char_buffer.render_chars(terminal)
        color_time = self.char_buffer.evolve_color(dt)
        evolve_time = self.char_buffer.evolve_state()

        return [spawn_time, render_time, color_time, evolve_time], f' ({len(self.char_buffer.char_states)}) [Asteroid Scene] (c)'

def get_scene_list():
    return {'asteroid': AsteroidScene}
