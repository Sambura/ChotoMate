import numpy as np
from PIL import Image

__print_progress_max_len__ = 0
def print_progress(counter, count, pre_message, post_message='', erase_overflown_text=True):
    global __print_progress_max_len__
    percentage = 100 * (counter + 1) / count
    msg = f'{pre_message} {counter + 1}/{count} ({percentage:3.1f}%) {post_message}'
    spaces = ''
    if erase_overflown_text:
        spaces = ' ' * max(0, __print_progress_max_len__ - len(msg))
        __print_progress_max_len__ = max(__print_progress_max_len__, len(msg))
    print(msg, spaces, end='\r')
    if counter == count - 1:
        print()
        __print_progress_max_len__ = 0

def quantize_video(frames: list, verbose: bool=True, threshold: int=127):
    "Turn a list of colored frames into a list of 1bit color frames. Optionally specify threshold separating two colors"
    if not verbose: # this is like 5% faster then the verbose version :)
        return list(np.mean(frames, axis=3) > threshold)

    q_frames = []
    for i, frame in enumerate(frames):
        if verbose: print_progress(i, len(frames), 'Quantizing video')
        q_frame = np.mean(frame, axis=2) > threshold
        q_frames.append(q_frame)

    return q_frames

def downscale_horrendous(img, new_w, new_h):
    "One of the most horrendous implementations of downscaling (but it does work! and fast!)"
    h, w = img.shape[:2]

    row_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    col_idx = (np.linspace(0, w - 1, new_w)).astype(int)

    return img[row_idx[:, None], col_idx]

def downscale_pil(img, new_w, new_h):
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_resized = pil_img.resize((new_w, new_h), Image.NEAREST)
    return np.array(pil_resized, dtype=bool)
