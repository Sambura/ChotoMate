from os import get_terminal_size
from sys import stdout
from time import perf_counter
from itertools import cycle
import random
import argparse

from functools import wraps

import numpy as np

SEQ_START = '\x1b['

ANSI_STYLE_RESET        = SEQ_START + '0m'
ANSI_INDEXED_COLOR_FG   = SEQ_START + '38;5;'
ANSI_TRUE_COLOR_FG      = SEQ_START + '38;2'
ANSI_TRUE_COLOR_BG      = SEQ_START + '48;2'
ANSI_NEW_BUFFER         = SEQ_START + '?1049h'
ANSI_LEAVE_BUFFER       = SEQ_START + '?1049l'
ANSI_CLEAR_SCREEN       = SEQ_START + '2J'
ANSI_CURSOR_HOME        = SEQ_START + 'H'
ANSI_SHOW_CURSOR        = SEQ_START + '?25h'
ANSI_HIDE_CURSOR        = SEQ_START + '?25l'
ANSI_SAVE_CURSOR        = SEQ_START + 's'
ANSI_RESTORE_CURSOR     = SEQ_START + 'u'
ANSI_FREEZE_BUFFER      = SEQ_START + '?2026h'
ANSI_DISPLAY_BUFFER     = SEQ_START + '?2026l'

# \x1b[5A   # up 5
# \x1b[3B   # down 3
# \x1b[10C  # right 10
# \x1b[4D   # left 4

def emit_sequence(seq): print(seq, end='')
def print_styled_text(text): print(text, end=ANSI_STYLE_RESET)

def _with_color(text, color):
    return f'{ANSI_TRUE_COLOR_FG};{color[0]};{color[1]};{color[2]}m{text}'

def _with_bg_color(text, color):
    return f'{ANSI_TRUE_COLOR_BG};{color[0]};{color[1]};{color[2]}m{text}'

def color8(r: int, g: int, b: int):
    return np.array([r, g, b], dtype=np.uint8)

def print_error(text, end='\n'):
    error_color = 1
    print_styled_text(f'{ANSI_INDEXED_COLOR_FG}{error_color}m{text}{ANSI_STYLE_RESET}{end}')

debug_log_buffer = []
def offscreen_debug_log(message: str, also_print_now: bool=False):
    debug_log_buffer.append(message)
    if also_print_now:
        print(message)

def dump_offscreen_debug_log():
    print(f'Dumping debug log ({len(debug_log_buffer)} messages):')
    for message in debug_log_buffer: print(message)

def smooth_value(prev, new, alpha:float=0.07):
    return (1 - alpha) * prev + alpha * new

def return_execution_time(has_return: bool=False):
    "Modify the function to return time elapsed for its execution. If has_return=True, returns tuple (result, elapsed) where result is the function's return value"
    def __wrapper(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            time_start = perf_counter()
            result = func(*args, **kwargs)
            elapsed = perf_counter() - time_start
            return (result, elapsed) if has_return else elapsed

        return _wrapper
    return __wrapper

class Terminal:
    width: int
    height: int

    def __init__(self, create_screen=True, double_buffering=True, hide_cursor=True):
        size = get_terminal_size()
        self.width = size.columns
        self.height = size.lines
        self.double_buffering = False
        self.new_buffer = False

        if create_screen:
            self.enter_screen()

        if double_buffering:
            self.init_double_buffering()

        if hide_cursor:
            self.hide_cursor()

    def enter_screen(self):
        emit_sequence(ANSI_NEW_BUFFER)
        emit_sequence(ANSI_CLEAR_SCREEN)
        emit_sequence(ANSI_CURSOR_HOME)
        self.new_buffer = True

    def exit_screen(self):
        emit_sequence(ANSI_LEAVE_BUFFER)
        self.new_buffer = False

    def init_double_buffering(self):
        emit_sequence(ANSI_FREEZE_BUFFER)
        self.double_buffering = True

    @return_execution_time()
    def flip(self):
        stdout.flush()

        if self.double_buffering:
            emit_sequence(ANSI_DISPLAY_BUFFER)
            emit_sequence(ANSI_FREEZE_BUFFER)

    def print_status(self, message):
        emit_sequence(ANSI_SAVE_CURSOR)
        self.set_cursor(0, self.height - 1)
        print_styled_text(_with_bg_color(_with_color(message, (230, 230, 140)), (0, 0, 0)))
        emit_sequence(ANSI_RESTORE_CURSOR)

    def set_cursor(self, x, y):
        # terminal cursor is apparently 1-based
        emit_sequence(f'{SEQ_START}{y + 1};{x + 1}H')

    def hide_cursor(self):
        emit_sequence(ANSI_HIDE_CURSOR)

    def show_cursor(self):
        emit_sequence(ANSI_SHOW_CURSOR)

    # context manager stuff
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.new_buffer:
            self.exit_screen()
        self.show_cursor()
        dump_offscreen_debug_log()

class CharState:
    x: int
    y: int
    color: np.ndarray
    tag: int

    def __init__(self, x: int, y: int, color, tag: int, evolve_func=None, data=(), next_evolve=0):
        self.x = x
        self.y = y
        self.color = color.astype(np.float32)
        self.tag = tag
        if evolve_func is None:
            evolve_func = CharState.__evolve_placeholder
        self.evolve_func = evolve_func
        self.next_evolve = next_evolve
        self.data = data

    def __evolve_placeholder(*args):
        return True, 1

    def evolve_state(self, char_buffer):
        keep, next_evolve = self.evolve_func(char_buffer, self, self.x, self.y, self.tag, *self.data)
        self.next_evolve = next_evolve
        return keep

class CharPallet:
    def __init__(self, pallet: str, char_width: int=1):
        self.pallet = pallet
        self.char_width = char_width

char_pallets = {
    None: None,
    'ABC': CharPallet(''.join(chr(x) for x in range(65, 65 + 26))),
    'binary': CharPallet('01'),
    'katakana': CharPallet('アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヰヱヲン', 2),
    'armenian': CharPallet('ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖՠաբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆևֈ'),
    'shavian': CharPallet('𐑐𐑑𐑒𐑓𐑔𐑕𐑖𐑗𐑘𐑙𐑚𐑛𐑜𐑝𐑞𐑟𐑠𐑡𐑢𐑣𐑤𐑥𐑦𐑧𐑨𐑩𐑪𐑫𐑬𐑭𐑮𐑯𐑰𐑱𐑲𐑳𐑴𐑵𐑶𐑷𐑸𐑹𐑺𐑻𐑼𐑽𐑾𐑿'),
}

class CharBuffer:
    default_pallet = char_pallets['ABC']

    char_pallet: str
    char_states: list[dict[int, CharState]]
    color_buffer: np.ndarray[np.uint8]
    width: int
    height: int

    def __init__(self, width: int, height: int, char_pallet: CharPallet):
        self.width = width
        self.height = height
        self.char_pallet = char_pallet
        self.char_states = []
        self.clear_char_buffer()
        self.color_buffer = np.zeros((height, width, 3), dtype=np.float32)
        self.tag_buffer = np.zeros((height, width), dtype=np.int_)
        self.color_evolvers = { 0: lambda colors, dt: colors } # default evolver (no color change)

    def clear_char_buffer(self, char=' '):
        "Reset entire character buffer to specified character (default: space)"
        self.char_buffer = [[char * self.char_pallet.char_width for _ in range(self.width)] for _ in range(self.height)]

    def get_random_char(self):
        return random.choice(self.char_pallet.pallet)

    @return_execution_time()
    def render_chars(self, terminal: Terminal):
        color_buffer_list = self.color_buffer.astype(np.uint8).tolist()
        for y in range(self.height):
            terminal.set_cursor(0, y)
            line = ''.join(_with_color(char, color_buffer_list[y][x]) for x, char in enumerate(self.char_buffer[y]))
            print(_with_bg_color('', color8(0, 0, 0)), end='')
            print_styled_text(line)

    @return_execution_time()
    def render_color_buffer(self, terminal: Terminal):
        color_buffer_list = self.color_buffer.astype(np.uint8).tolist()
        for y in range(self.height):
            terminal.set_cursor(0, y)
            print_styled_text(''.join(_with_bg_color(' ', color_buffer_list[y][x]) for x in range(self.width)))

    def update_char_position(self, char_state: CharState, character: str):
        self.char_buffer[char_state.y][char_state.x] = character
        self.color_buffer[char_state.y][char_state.x] = char_state.color
        self.tag_buffer[char_state.y][char_state.x] = char_state.tag

    def add_char_state(self, char_state: CharState, update_position: bool=True):
        self.char_states.append(char_state)
        if update_position:
            self.update_char_position(char_state, self.get_random_char())

    def print_line_to_char_buffer(self, text: str, x: int, y: int, color=None, centered=False):
        "print a single line to the char buffer at the specified coordinates"
        if y < 0 or y >= self.height: return
        if centered: x = x - len(text) // 2

        if x < 0:
            text = text[-x:]
            x = 0

        for c in text:
            if x >= self.width:
                return

            self.char_buffer[y][x] = c
            if color is not None:
                self.color_buffer[y, x] = color

            x += 1

    def set_color_buffer_evolver(self, tag: int, evolver, evolve_alive=True, evolve_dead=True):
        if tag < 0:
            raise ValueError(f'Tag should be positive, got: {tag}. To set an evolver exclusively for dead characters, use evolve_alive=False')

        dead_tag = -tag - 1
        if evolver is None:
            self.color_evolvers.pop(tag, None)
            self.color_evolvers.pop(dead_tag, None)
        else:
            if evolve_alive:
                self.color_evolvers[tag] = evolver
            if evolve_dead:
                self.color_evolvers[dead_tag] = evolver

    @return_execution_time()
    def evolve_color(self, delta_time):
        self.color_buffer = np.sum(
            [(self.tag_buffer == tag)[:, :, np.newaxis] * evolver(self.color_buffer, delta_time) for tag, evolver in self.color_evolvers.items()],
            axis=0
        ).astype(np.float32)

    @return_execution_time()
    def evolve_state(self):
        alive_mask = self.tag_buffer > 0                                        # kill positive tags: 1 -> -2, 2 -> -3, etc.
        footprint_mask = self.tag_buffer < -1                                   # preserve all values below -1 as is
        blank_mask = np.logical_not(np.logical_or(alive_mask, footprint_mask))  # set everything else to -1

        self.tag_buffer = footprint_mask * self.tag_buffer - 1 * blank_mask - (self.tag_buffer + 1) * alive_mask

        ts = perf_counter()
        for i in range(len(self.char_states))[::-1]:
            char_state = self.char_states[i]

            if ts < char_state.next_evolve:
                self.tag_buffer[char_state.y,char_state.x] = char_state.tag
                continue

            keep = self.char_states[i].evolve_state(self)

            if not keep:
                del self.char_states[i]

class Scene:
    def do_frame(self, terminal: Terminal, delta_time: float):
        return [], ' Dummy scene rendering now...'

    def register_arg_parser(subparser: argparse.ArgumentParser):
        subparser.description = 'This scene does not have configurable parameters'

class CharBufferScene(Scene):
    def __init__(self, width: int, height: int, char_pallet: CharPallet=None):
        if char_pallet is None: char_pallet = CharBuffer.default_pallet
        adjusted_width = width // char_pallet.char_width

        self.width = adjusted_width
        self.height = height
        self.char_buffer = CharBuffer(adjusted_width, height, char_pallet)

class TestScene(CharBufferScene):
    def __init__(self, width: int, height: int, char_pallet: CharPallet, args: argparse.Namespace):
        super().__init__(width, height, char_pallet)

        midpoint = min(width, height) // 2
        offset = min(width, height) // 5
        self.spinner = CharState((midpoint + offset) * 2, midpoint - offset, color8(255, 255, 255), 0)
        self.spinner_char = cycle('/-\\|')
        self.next_spin = 0

        for y in range(height):
            string = f'{y}'
            if char_pallet is not None:
                string = char_pallet.pallet[y % len(char_pallet.pallet)]

            self.char_buffer.print_line_to_char_buffer(string, 2 * y, y, color8(255, 255, 255))

    def do_frame(self, terminal: Terminal, dt: float):
        if self.next_spin <= perf_counter():
            self.char_buffer.update_char_position(self.spinner, next(self.spinner_char))
            self.next_spin = perf_counter() + 0.07

        render_time = self.char_buffer.render_chars(terminal)

        return [render_time], f' [Test]'

class ColorEvolvers:
    def exponential_fade(fade_rate: float=0.97):
        "Multiplies color by fade_rate upon invocation. **Warning**: fade speed depends on FPS"
        def evolver(color_buffer: np.ndarray[float], delta_time: float):
            return color_buffer * fade_rate

        return evolver

    def power_fade(fade_power: float=1.01, uniform_channel_fade: bool=True, method: str=None):
        "Raises color (in range [0..1]) to fade_power. **Warning**: fade speed depends on FPS"
        if not uniform_channel_fade and method is not None:
            raise ValueError('method should not be specified unless uniform_channel_fade is True')

        def evolver_base(color_buffer: np.ndarray[float], delta_time: float):
            return 256 * ((color_buffer / 256) ** fade_power)

        if method is None: method = 'max'
        methods = {'max': np.max, 'min': np.min, 'mean': np.mean}
        if method not in methods:
            raise ValueError(f'Unknown method: {method}')
        func = methods[method.lower()]

        def evolver_uniform(color_buffer: np.ndarray[float], delta_time: float):
            brightness = func(color_buffer, axis=2)
            normalized = brightness / 256 # divide by 256 to ensure all values are *less* than one 1
            fade_amount = 1 - (normalized - normalized ** fade_power)

            return fade_amount[:, :, np.newaxis] * color_buffer

        return evolver_uniform if uniform_channel_fade else evolver_base

    def preserve_color(color_buffer: np.ndarray[float], delta_time: float):
        "Keeps the same color as the last frame"
        return color_buffer

def positive_float(value):
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError(f'{x} should be positive')
    return x

def custom_float(min_value=None, max_value=None, min_inclusive=True, max_inclusive=True):
    def _checker(value):
        x = float(value)
        if min_value is not None and x < min_value or (x == min_value and not min_inclusive):
            verb = 'greater than or equal' if min_inclusive else 'greater than'
            raise argparse.ArgumentTypeError(f'argument should be {verb} {min_value}')
        if max_value is not None and x > max_value or (x == max_value and not max_inclusive):
            verb = 'less than or equal' if min_inclusive else 'less than'
            raise argparse.ArgumentTypeError(f'argument should be {verb} {max_value}')
        return x

    return _checker

def float01(value):
    x = float(value)
    if not (0.0 <= x <= 1.0):
        raise argparse.ArgumentTypeError(f'{x} should be in range [0.0, 1.0]')
    return x

def positive_int(value):
    x = int(value)
    if x <= 0:
        raise argparse.ArgumentTypeError(f'{x} should be positive')
    return x

def color_arg(value: str):
    hex_color = value.lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError(f'expected 6 hex characters')
    return color8(*[int(hex_color[i:i+2], 16) for i in (0, 2, 4)])
