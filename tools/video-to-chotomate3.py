import sys
import numpy as np
import struct
from pathlib import Path
from math import ceil

sys.path.append(str(Path(__file__).resolve().parent.parent))

from media.common import decode_video, quantize_video, downscale_horrendous, print_progress
from common.base import ANSI_SAVE_CURSOR, ANSI_RESTORE_CURSOR, emit_sequence

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

def get_max_encode_size(frame):
    height, width = frame.shape
    return 1 + 4 + 4 * height * width

def get_packed_count(count):
    if count == 0:
        return '', 0, 0
    if count < 257: # we don't want to be able to encode 0
        return 'B', 1, count - 1
    elif count < 65536 + 257:
        return 'H', 2, count - 257
    else:
        return 'I', 4, count # do not manipulate count since it fits either way, don't wast compute

def encode_descriptor_012(frame_type, base_type, om_format, direction):
    assert 0 <= frame_type < 8
    assert 0 <= base_type < 4
    assert direction == 0 or direction == 1
    om_map = {'': 0, 'B': 1, 'H': 2, 'I': 3}
    assert om_format in om_map
    om = om_map[om_format]

    descriptor = base_type | (om << 2) | (direction << 4) | (frame_type << 5)
    assert 0 <= descriptor <= 256
    return descriptor

def pack_short_value(value, buffer, offset):
    assert 0 < value < 4294967296 # positive and can fit in uint32

    if value < 255: # deltas: 0..254; byte value: 0..253
        struct.pack_into('<B', buffer, offset, value - 1)
        return 1
    elif value < 65536 + 255: # deltas: 255..65535 + 255
        struct.pack_into('<BH', buffer, offset, 254, value - 255)
        return 3
    else:
        struct.pack_into('<BI', buffer, offset, 255, value)
        return 5

invalid_encoding = range(5000 * 5000 * 100000 * 2 * 4) # just very long encoding (don't mind the numbers)

def get_encoders_0(frame, prev_frame, diff_based, shifted_base, quick_eval=False):
    def _do_encode(sparse_pixels, base_type, size_limit):
        format, format_size, count_value = get_packed_count(len(sparse_pixels))
        initial_pixel = int(0 if len(sparse_pixels) == 0 else sparse_pixels[0] == 0)
        descriptor = encode_descriptor_012(0, base_type, format, initial_pixel)
        if format == '':
            return struct.pack('<B', descriptor)

        data_offset = 1 + format_size
        if quick_eval or data_offset + len(sparse_pixels) >= size_limit:
            return invalid_encoding
        buffer = bytearray(data_offset + len(sparse_pixels) * 5) # max buffer size, truncate later
        struct.pack_into(f'<B{format}', buffer, 0, descriptor, count_value)

        offset_origin = -initial_pixel
        for offset in sparse_pixels:
            data_offset += pack_short_value(offset - offset_origin, buffer, data_offset)
            offset_origin = offset

        return buffer[:data_offset]

    def _base_black(max_size: int):
        sparse_pixels = np.flatnonzero(frame)
        return _do_encode(sparse_pixels, 0, max_size)

    def _base_white(max_size: int):
        sparse_pixels = np.flatnonzero(frame == 0)
        return _do_encode(sparse_pixels, 1, max_size)

    def _base_frame(max_size: int):
        if prev_frame is None: return invalid_encoding
        sparse_pixels = np.flatnonzero(frame != prev_frame)
        return _do_encode(sparse_pixels, 3 if shifted_base else 2, max_size)

    return [_base_frame] if diff_based else [_base_black, _base_white, _base_frame]

def _cm2__get_frame_deltas(flat_frame: np.ndarray, current_value: bool, current_count: int, total_size: int):
    flat_delta = np.zeros_like(flat_frame)
    flat_delta[0]    = current_value != flat_frame[0]
    flat_delta[1:]   = flat_frame[1:] != flat_frame[:-1]
    # flat_delta contains a 1 at every index where value is different then the last

    switch_indices = np.nonzero(flat_delta)[0] # get indices where value switches

    # if len(switch_indices) > 0:
    #     last_indices = np.zeros_like(switch_indices)
    #     last_indices[0] = -current_count
    #     last_indices[1:] = switch_indices[:-1]
    #     last_index = switch_indices[-1]
    #     current_count = 0

    #     # vectorized: delta = switch_indices[i] - switch_indices[i - 1]; add current_count to the first delta
    #     deltas = (switch_indices - last_indices).tolist()
    # else:
    #     last_index = 0
    #     deltas = []

    deltas = []
    last_index = 0
    for index in switch_indices:
        count = index - last_index + current_count
        deltas.append(count)
        last_index = index
        current_count = 0

    current_count += total_size - last_index
    current_value = current_value if len(switch_indices) % 2 == 0 else not current_value
    return deltas, current_value, current_count

def get_encoders_3(frame, prev_frame, shifted_base, diff_based):
    if shifted_base:
        return invalid_encoding
    height, width = frame.shape

    def _do_encode(frame, is_diff, direction, size_limit):
        initial_value = frame[0,0]
        flat_frame = frame.ravel() if direction == 0 else frame.ravel(order='F')
        deltas, _, _ = _cm2__get_frame_deltas(flat_frame, initial_value, 0, width * height)

        format, format_size, count_value = get_packed_count(len(deltas))
        descriptor = encode_descriptor_012(3, (is_diff << 1) | initial_value, format, direction)

        data_offset = 1 + format_size
        if data_offset + len(deltas) >= size_limit:
            return invalid_encoding
        buffer = bytearray(data_offset + len(deltas) * 5) # max possible size, truncate later
        struct.pack_into(f'<B{format}', buffer, 0, descriptor, count_value)

        for delta in deltas:
            data_offset += pack_short_value(delta, buffer, data_offset)

        return buffer[:data_offset]

    if not diff_based:
        encoders = [lambda m: _do_encode(frame, 0, 0, m), lambda m: _do_encode(frame, 0, 1, m)]
    else:
        encoders = []
    if prev_frame is not None:
        diff = np.logical_xor(frame, prev_frame)
        encoders += [lambda m: _do_encode(diff, 1, 0, m), lambda m: _do_encode(diff, 1, 1, m)]

    return encoders

def get_encoders_5(frame, prev_frame, shifted_base, diff_based):
    def pack_special_value(value, byte_count, max_bit_masks, buffer, offset):
        max_byte = {1: 127, 2: 127, 3: 63, 4: 63, 5: 31, 6: 31, 7: 31, 8: 31}[max_bit_masks]

        assert 0 <= value < 4294967296 # positive and can fit in uint32
        # TODO actually encode byte_count and adjust constants
        if value < max_byte: # deltas: 0..254; byte value: 0..253
            struct.pack_into('<B', buffer, offset, value)
            return 1
        elif value < 65536 + max_byte: # deltas: 255..65535 + 255
            struct.pack_into('<BH', buffer, offset, 254, value - max_byte)
            return 3
        else:
            struct.pack_into('<BI', buffer, offset, 255, value)
            return 5

    def _do_encode(sparse_pixels, base_type, size_limit, direction):
        byte_offsets = sparse_pixels // 8
        max_bit_masks = 2
        # index_deltas = sparse_pixels[1:] - sparse_pixels[:-1] - 1
        # segment_boundaries = np.flatnonzero(index_deltas)

        # # more or less minimal encoding size
        # if (len(segment_boundaries) + 1) // 8 + 2 >= size_limit:
        #     return invalid_encoding

        encode = []

        offset_origin = 0
        for i in range(len(sparse_pixels)):
            byte_offset = byte_offsets[i]
            if byte_offset < offset_origin:
                continue

            byte_start = byte_offset * 8
            range_end = min(len(sparse_pixels), i + max_bit_masks * 8)
            affected_pixels = [sparse_pixels[j] - byte_start for j in range(i, range_end) if byte_offsets[j] - byte_offset < max_bit_masks]
            byte_count = int(ceil(affected_pixels[-1] / 8))
            actual_offset = byte_offset - offset_origin

            encode.append((actual_offset, byte_count, affected_pixels))
            offset_origin = byte_offset + byte_count

        format, format_size, count_value = get_packed_count(len(encode))
        descriptor = encode_descriptor_012(5, base_type, format, direction)

        # print(f'\nEncode 5: {len(encode)} bit mask segments')
        data_offset = 1 + format_size
        if data_offset + len(encode) + np.sum([byte_count for _, byte_count, _ in encode]) >= size_limit:
            return invalid_encoding
        buffer = bytearray(data_offset + len(encode) * (5 + max_bit_masks)) # max buffer size, truncate later
        struct.pack_into(f'<B{format}', buffer, 0, descriptor, count_value)

        for offset, byte_count, bits in encode:
            # TODO: actually pack bits in
            data_offset += pack_special_value(offset, byte_count, max_bit_masks, buffer, data_offset)
            if byte_count == 1:
                struct.pack_into('<B', buffer, data_offset, 0)
            else:
                struct.pack_into('<BB', buffer, data_offset, 0, 0)
            data_offset += byte_count

        return buffer[:data_offset]

    def _base_black(max_size: int, order: str):
        sparse_pixels = np.nonzero(frame.ravel(order=order))[0]
        return _do_encode(sparse_pixels, 0, max_size, int(order == 'F'))

    def _base_white(max_size: int, order: str):
        sparse_pixels = np.nonzero((frame == 0).ravel(order=order))[0]
        return _do_encode(sparse_pixels, 1, max_size, int(order == 'F'))

    def _base_frame(max_size: int, order: str):
        if prev_frame is None: return invalid_encoding
        sparse_pixels = np.nonzero((frame != prev_frame).ravel(order=order))[0]
        return _do_encode(sparse_pixels, 3 if shifted_base else 2, max_size, int(order == 'F'))

    return [lambda m: _base_frame(m, 'C'), lambda m: _base_frame(m, 'F')] if diff_based else [
        lambda m: _base_black(m, 'C'), lambda m: _base_white(m, 'C'), lambda m: _base_frame(m, 'C'),
        lambda m: _base_black(m, 'F'), lambda m: _base_white(m, 'F'), lambda m: _base_frame(m, 'F')
    ]

def get_encoders_6(frame, base_frame, shifted_base, diff_based):
    def _do_encode(frame, base_type, block_size_selector, size_limit):
        block_size = 8 if block_size_selector else 4

        bytes_per_block = (block_size * block_size) // 8
        block_format = '<' + 'B' * bytes_per_block
        assert block_size * block_size == bytes_per_block * 8
        blocks_x, blocks_y = int(ceil(frame.shape[1] / block_size)), int(ceil(frame.shape[0] / block_size))
        blocks = []
        for block_y in range(blocks_y):
            for block_x in range(blocks_x):
                oy, ox = block_y * block_size, block_x * block_size
                block_id = block_x + block_y * blocks_x
                block = frame[oy:oy + block_size, ox:ox + block_size]
                if np.count_nonzero(block) == 0:
                    continue

                blocks.append((block_id, np.flatnonzero(block)))

        format, format_size, count_value = get_packed_count(len(blocks))
        descriptor = encode_descriptor_012(6, base_type, format, block_size_selector)

        data_offset = 1 + format_size
        if data_offset + len(blocks) * (bytes_per_block + 1) >= size_limit:
            return invalid_encoding
        buffer = bytearray(data_offset + len(blocks) * (bytes_per_block + 5)) # max buffer size, truncate later
        struct.pack_into(f'<B{format}', buffer, 0, descriptor, count_value)

        offset_origin = -1
        for block_id, block_diff in blocks:
            actual_offset = block_id - offset_origin
            data_offset += pack_short_value(actual_offset, buffer, data_offset)
            struct.pack_into(block_format, buffer, data_offset, *([0] * bytes_per_block))
            data_offset += bytes_per_block
            offset_origin = block_id

        return buffer[:data_offset]

    def _base_black(max_size: int, block_size: int):
        return _do_encode(frame, 0, block_size, max_size)

    def _base_white(max_size: int, block_size: int):
        pixels = frame == 0
        return _do_encode(pixels, 1, block_size, max_size)

    def _base_frame(max_size: int, block_size: int):
        if base_frame is None: return invalid_encoding
        pixels = frame != base_frame
        return _do_encode(pixels, 3 if shifted_base else 2, block_size, max_size)

    return [lambda m: _base_frame(m, 0), lambda m: _base_frame(m, 1)] if diff_based else [
        lambda m: _base_black(m, 0), lambda m: _base_black(m, 1),
        lambda m: _base_white(m, 0), lambda m: _base_white(m, 1),
        lambda m: _base_frame(m, 0), lambda m: _base_frame(m, 1)
    ]

def plot_frame_and_diff(frame, prev_frame, frame_index, size, code):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap([
        "red",    # -1
        "white",  # 0
        "green",  # 1
    ])

    norm = BoundaryNorm(
        boundaries=[-1.5, -0.5, 0.5, 1.5],
        ncolors=cmap.N
    )

    diff = np.zeros_like(frame) if prev_frame is None else frame.astype(int) - prev_frame.astype(int)
    if np.count_nonzero(frame) == 0 and np.count_nonzero(diff) == 0:
        print('Skipping empty frame...')
        return False

    fig, (axl, axc, axr) = plt.subplots(1, 3)
    axl.imshow(frame)
    axc.imshow(np.zeros_like(frame) if prev_frame is None else prev_frame)
    axr.imshow(diff, cmap=cmap, norm=norm)
    axl.set_title(f'Frame {frame_index}')
    axc.set_title(f'Prev frame')
    axr.set_title(f'Diff {frame_index}')
    fig.suptitle(f'Frame type {code}; size: {size} bytes')

    plt.show(block=False)
    return True

def offset_frame(frame, x, y):
    padding = max(abs(x), abs(y))
    src_frame = np.pad(frame, padding, mode='edge')

    src_y0, src_y1 = padding + y, padding + y + frame.shape[0]
    src_x0, src_x1 = padding + x, padding + x + frame.shape[1]
    new_frame = src_frame[src_y0:src_y1, src_x0:src_x1]
    return new_frame

def get_shift_frame_encoders(frame, base_frame, shift_space):
    shift_encoders = []
    if base_frame is None:
        return shift_encoders

    for offset_x, offset_y in shift_space:
        new_base_frame = offset_frame(base_frame, offset_x, offset_y)

        encoders = get_encoders_0(frame, new_base_frame, True, True) +\
                   get_encoders_5(frame, new_base_frame, True, True) +\
                   get_encoders_6(frame, new_base_frame, True, True)

        shift_encoders += encoders

    # TODO: these encoders should also append an additional byte per frame but oh well
    return shift_encoders

def encode_frame(frame, prev_frame, next_frame, shift_space, only_diff_based: bool, existing_encode=invalid_encoding):
    encoders =  get_encoders_0(frame, prev_frame, only_diff_based, False, quick_eval=True) +\
                get_encoders_3(frame, prev_frame, only_diff_based, False) +\
                get_encoders_0(frame, prev_frame, only_diff_based, False, quick_eval=False) +\
                get_encoders_5(frame, prev_frame, only_diff_based, False) +\
                get_encoders_6(frame, prev_frame, only_diff_based, False) +\
                get_shift_frame_encoders(frame, prev_frame, shift_space)

    best_encode = existing_encode
    for encoder in encoders:
        new_encode = encoder(len(best_encode))
        if len(new_encode) < len(best_encode):
            best_encode = new_encode

            if len(best_encode) == 1:
                break

    if len(best_encode) > get_max_encode_size(frame):
        raise RuntimeError(f'Encode error: got size {len(best_encode)} with max size {get_max_encode_size(frame)}')

    return best_encode

# CHOTOMATE3 FORMAT DRAFT
# 
# Frame format:
# 1 Byte: frame metadata:
#       T T T P4 P3 P2 P1 P0:
#           - 3 bits (T): frame type
#           - 5 bits (P): parameter

# Frame types:
#  - type 0:
#       sparse pixels. Parameter [1,0] specifies the base frame info. Parameter [3,2] specifies data count. Parameter [4] specifies the initial offset:
#           Paramter [4]      
#               offset tis 0 if 0, and -1 if 1 (-1 is needed since pixel offsets are packed in a way where they cannot encode a zero):
#           Parameter [1,0]:
#               00: fill frame with black
#               01: fill frame with white
#               10: base on previous frame
#               11: base on next frame
#       OrderOfMagnitude (2 bits):
#           0: no additional data follows. the frame is just a solid color
#           1: the following uint8 specifies the number of sparse pixel entries
#           2: the following uint16 specifies the number of sparse pixel entries
#           3: the following uint32 specifies the number of sparse pixel entries
#       Next follow SparsePixelEntryCount packed numbers, each number represent an offset into a flattened frame where a pixel color should be altered. Offset's origin is previous offset
#  - type 1 [not useful, discarded]:
#       full segments. Parameter [1,0] specifies the base frame info. Parameter [3,2] specifies data count. Parameter [4] specifies segment direction
#           Parameter [4]: 0: horizontal, 1: vertical
#       Order of magnitude: same as type 0, resulting value is SegmentCount
#       Next follow SegmentCount pairs if uint32 numbers. Each number is a pixel offset into a flattened frame.
#       Each segment flips all pixels from segment_start offset to segment_end offset, inclusive
#  - type 2 [almost never useful, discarded]
#       short segments. Descriptor: same as type 1
#       After SegmentCount follow SegmentCount pairs of uint16 numbers. the meaning is as follows:
#           segment_offset: offset counting from the previous segment end
#           segment_size:   count of pixels in this segment
#           first segment starts at offset 0. The second segment starts at first_segment_offset + first_segment_count
#           each segment inverts the base frame colors
#           if the required offset does not fit in uint16 number, use a segment (65535, 0) to move the pointer 65535 pixels forward
#  - type 3:
#       chotomate original encoding. Parameter [0]: initial color; Parameter [1]: use last frame; parameter [3,2] specifies data count; Parameter [4] specifies ravel direction
#       next follows delta count, encoded as described by OrderOfMagnitude
#       Next follow DeltaCount data entries encoded with chotomate2 method
#       LAST FRAME MODE: When Parameter [1] is set to 1, the encoder will encode the diff with the previous frame instead of the current frame. Parameter [0] means: 0: start by no diff, 1: start with diff
#   - type 4: [terrible idea, discarded]
#       chotomate original encoding, but per 16x16 blocks. Parameter [0]: initial color, applies to each block beginning; Parameter [1]: use last frame; parameter [3,2] specifies data count; Parameter [4] specifies block ravel direction
#       after descriptor follows value BlockCount
#       next follow BlockCount entries:
#           packed block offset
#           block encoding struct
#       each block encoding struct has the following structure:
#           segment count: 1 byte (1-256)
#           SegmentCount offsets: regular chotomate encoding, 1 byte each
#   - type 5:
#       sparse bitmask encoding. Parameter [1,0] specifies the base frame info. Parameter [3,2] specifies data count. Parameter [4] specifies frame ravel direction
#           EntryCount diff structs:
#           Diff struct:
#               special packed offset*
#               bitmask diff: 1-2 bytes
#           *special packed offset:
#               first bit: size of bitmask diff: 0 is one byte, 1 is two bytes
#               rest 7 bits: packed offset
#    - type 6:
#       similar to type 5 but using 4x4 pixel blocks instead of 1-2 bytes strips
#
#  - other ideas: rect frame / combo frame (combination of frame types) / chotomate2-like encoding

def is_valid_base_next_frame(encoded_frame):
    frame_type = (encoded_frame[0] & 228) >> 5
    base_type = encoded_frame[0] & 3

    ok = ((frame_type == 3 or frame_type == 4) and base_type < 2) or base_type != 2
    return ok

def get_frame_code(encoded_frame):
    frame_type = (encoded_frame[0] & 228) >> 5                              # bits 7,6,5
    base_type = encoded_frame[0] & 3                                        # bits 0,1
    direction = (encoded_frame[0] & 16) >> 4                                # bit 4
    data_om = (encoded_frame[0] & 12) >> 2                                  # bits 2, 3

    base_type_code = { 0: 'B', 1: 'W', 2: '>', 3: '~'}[base_type]     
    if frame_type == 3 or frame_type == 4: base_type_code = { 0: 'x', 1: 'x', 2: '>', 3: '>'}[base_type]
    direction_code = 'H' if direction == 0 else 'V'
    if frame_type == 0: direction_code = ''
    om_code = {0: 'Z', 1: 'B', 2: 'H', 3: 'I'}[data_om]

    return f'{frame_type}{base_type_code}-{om_code}{direction_code}'

def show_preview(frame, w, h):
    preview = downscale_horrendous(frame, w, h)
    emit_sequence(ANSI_SAVE_CURSOR)
    print() # newline

    for y in range(h):
        line = ''.join('#' if x else ' ' for x in preview[y, :])
        print(line)

    print() # newline
    emit_sequence(ANSI_RESTORE_CURSOR)

def encode_frames_pass(frames, encoded_frames, max_shift=4, pruned_shift_space=True):
    encodings = []
    total_size = 0
    frame_code_stats = {}

    shift_space = [(x, y) for x in range(-max_shift, max_shift + 1) for y in range(-max_shift, max_shift + 1)]
    if pruned_shift_space:
        shift_space = [(x, y) for x, y in shift_space if abs(x) + abs(y) <= max_shift and (x != 0 or y != 0)]

    preview_height = 15
    preview_width = round(2 * frames[0].shape[1] / frames[0].shape[0] * preview_height)

    for i, (frame, next_eframe) in enumerate(zip(frames, encoded_frames[1:] + [None])):
        prev_frame = None if i == 0 else frames[i - 1]
        # we can use next frame if the next frame is *not* generated from this frame
        # note that if a frame is not generated from the previous frame, this fact is not going to change on the next pass
        next_frame = None if next_eframe is None or not is_valid_base_next_frame(next_eframe) else frames[i + 1]
        encoded_frame = encode_frame(frame, prev_frame, next_frame, shift_space, False)
        current_size = len(encoded_frame)
        total_size += current_size
        encodings.append(encoded_frame)

        frame_code = get_frame_code(encoded_frame)
        stat = frame_code_stats.get(frame_code, 0)
        frame_code_stats[frame_code] = stat + 1

        show_preview(frame, preview_width, preview_height)
        print_progress(i, len(frames), 'Encoding frames', f'Current: {frame_code}; {current_size} bytes, avg: {total_size/(i+1):0.1f} bytes')

    show_preview(np.zeros(1000, 1000), preview_width, preview_height) # clear
    print('Pass complete. Frame stats:')
    for frame_code, count in frame_code_stats.items():
        print(f'\tFrame {frame_code}: {count} frames encoded')
    print()

    return encodings

def encode_interactive(frames, encoded_frames):
    import matplotlib.pyplot as plt

    # 21x21 square, meaning up to 10 pixels offset in either direction
    # precompute_spiral = [x for x, _ in zip(spiral(), range(440))]
    # precompute_spiral = [(x, y) for x, y in precompute_spiral if -7 <= x <= 8 and -7 <= y <= 8]
    precompute_spiral = [(0, 1), (0, -1), (1, 0), (-1, 0)] # dummy values :)

    for i, (frame, next_eframe) in enumerate(zip(frames, encoded_frames[1:] + [None])):
        prev_frame = None if i == 0 else frames[i - 1]
        # we can use next frame if the next frame is *not* generated from this frame
        # note that if a frame is not generated from the previous frame, this fact is not going to change on the next pass
        next_frame = None if next_eframe is None or not is_valid_base_next_frame(next_eframe) else frames[i + 1]
        encoded_frame = encode_frame(frame, prev_frame, next_frame, False)
        current_size = len(encoded_frame)
        frame_code = get_frame_code(encoded_frame)

        print(f'Encoding frames {i + 1}/{len(frames)} Current: {frame_code}; {current_size} bytes')

        # ok_frame = plot_frame_and_diff(frame, prev_frame, i, current_size, frame_code)
        ok_frame = np.count_nonzero(frame) != 0 and (prev_frame is None or np.count_nonzero(frame != prev_frame) != 0)

        orig_size = current_size
        prompt = 'Try re-encoding frame? y/n'
        stage = 0
        while True and ok_frame:
            # cmd = input(prompt)
            cmd = 'y'
            if stage == 0 and cmd == 'n':
                break
            if stage == 0 and cmd == 'y':
                stage = 1
                prompt = 'specify offset as `x y`'
                continue
            if stage == 0:
                continue
            cmd = 'auto'
            if cmd == 'continue':
                break
            if cmd == 'auto':
                plt.close()
                best_encode = encoded_frame
                for i, (offset_x, offset_y) in enumerate(precompute_spiral):
                    print(f'{i+1}/{len(precompute_spiral)}', end='\r')
                    new_frame = offset_frame(prev_frame, offset_x, offset_y)
                    encoded_frame = encode_frame(frame, new_frame, next_frame, True, existing_encode=best_encode)
                    frame_code = get_frame_code(encoded_frame)
                    if len(best_encode) > len(encoded_frame):
                        best_encode = encoded_frame
                        print(f'Found better encoding: {offset_x}; {offset_y}: {orig_size} -> {len(best_encode)} ({frame_code}) [{len(best_encode) - orig_size}]')

                print(f'Auto scan complete')
                break

            offset_x, offset_y = [int(x) for x in cmd.split(' ')]
            new_frame = offset_frame(prev_frame, offset_x, offset_y)

            plt.close()
            encoded_frame = encode_frame(frame, new_frame, next_frame, False)
            current_size = len(encoded_frame)
            frame_code = get_frame_code(encoded_frame)

            print(f'Re-encoded frame {i + 1}/{len(frames)} with offset {offset_x}, {offset_y} Current: {frame_code}; {current_size} bytes')
            plot_frame_and_diff(frame, new_frame, i, current_size, frame_code)
        
        if ok_frame:
            plt.close()

def vertical_encoding_test(frames):
    frame_space = np.array(frames) # (frames, 360, 480, 1)

    frame_space_flat = frame_space.reshape((len(frames), -1))
    strands_count = frame_space_flat.shape[1]
    print(f'Strand count: {strands_count}')
    
    total_deltas = []
    for i in range(strands_count):
        flat_frame = frame_space_flat[:, i]
        deltas, _, current_count = _cm2__get_frame_deltas(flat_frame, False, 0, len(frames))
        deltas.append(current_count)
        total_deltas += deltas
        print(f'Deltas count {i + 1}/{strands_count}: {len(deltas)}')

    print(f'Avg delta count per strand: {len(total_deltas) / strands_count:.1f} deltas, mean delta: {np.mean(total_deltas):0.1f}; lower encoding bound: {(len(total_deltas) + strands_count)/1024:0.1f} KB')

def process_video(path):
    frames, timestamps = decode_video(path)
    frames = quantize_video(frames)

    # vertical_encoding_test(frames)
    # return

    # general info
    height, width = frames[0].shape
    frames_count = len(frames)

    # compute time data
    ts_deltas, ts_resolution = encode_timestamps(timestamps)

    # compute frame data
    pass_count = 1
    encoded_frames = [None] * frames_count
    for _pass in range(pass_count):
        print(f'Encoding pass {_pass + 1}')
        encoded_frames = encode_frames_pass(frames, encoded_frames)
        # encoded_frames = encode_interactive(frames, encoded_frames)

    print('test encode done, exiting')
    exit(0)

    # write out
    out_path = f'{str(Path(path).stem)}.chotomate3'
    with open(out_path, 'wb') as f:
        print('Writing header...')
        header = struct.pack('<HH I I B', width, height, frames_count, len(frame_deltas), ts_resolution)
        f.write(header)

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
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <video_path>")
        sys.exit(2)

    process_video(sys.argv[1])
