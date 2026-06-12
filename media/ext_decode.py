import cv2
import numpy as np

def decode_video(path: str, verbose: bool=True, frame_limit: int=-1) -> tuple[list[np.ndarray], list[float]]:
    "Decode frame and timestamp info from a video file at `path`, up to `frame_limit` frames. Returns list of frames and list of timestamps"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')

    if frame_limit < 0: frame_limit = float('inf')

    frames = []
    timestamps = []

    while len(frames) < frame_limit:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            if len(frames) == 0:
                raise RuntimeError("No frames decoded from video.")
            break

        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamps.append(ts_ms)
        frames.append(frame)
        if verbose: print(f'Decoding frame {len(frames)} ({ts_ms/1000:0.1f} s)', end='\r')

    if verbose: print()
    return frames, timestamps
