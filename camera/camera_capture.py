from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

import cv2


from picamera2 import Picamera2
import time

class ThreadedCamera:
    def __init__(self, camera_index=0, width=640, height=480):
        self.picam2 = Picamera2()

        config = self.picam2.create_preview_configuration(
            main={"size": (1920,1080)}
        )

        self.picam2.configure(config)
        self.picam2.start()

        time.sleep(2)

        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        thread = threading.Thread(target=self._update, daemon=True)
        thread.start()
        return self

    def _update(self):
        while not self.stopped:
            frame = self.picam2.capture_array()

            with self.lock:
                self.frame = frame
                self.grabbed = True

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None

            return self.grabbed, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.picam2.stop()

def iter_image_paths(source: str) -> Iterator[Path]:
    path = Path(source)
    if path.is_file():
        yield path
        return

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for image_path in sorted(path.rglob("*")):
        if image_path.suffix.lower() in image_exts:
            yield image_path


def open_video(source: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return capture
