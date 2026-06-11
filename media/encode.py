import numpy as np

def compress_color_stripes_continuous(flat_frame: np.ndarray, current_value: bool, current_count: int, total_size: int):
    flat_delta = np.zeros_like(flat_frame)
    flat_delta[0]    = np.logical_xor(current_value, flat_frame[0])
    flat_delta[1:]   = np.logical_xor(flat_frame[1:], flat_frame[:-1])

    deltas = []
    switch_indices = np.nonzero(flat_delta)[0]
    last_index = 0
    for index in switch_indices:
        count = index - last_index + current_count
        deltas.append(count)
        last_index = index
        current_count = 0

    current_count += total_size - last_index
    current_value = current_value if len(switch_indices) % 2 == 0 else not current_value
    return deltas, current_value, current_count
