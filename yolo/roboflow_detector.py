from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from yolo.detector import Detection


ROBOFLOW_API_URL = "https://detect.roboflow.com"

ROBOFLOW_CLASS_MAP = {
    "Stain": 0,
    "hole": 2,
    "needle line": 1,
}

LOCAL_CLASS_NAMES: dict[int, str] = {
    0: "stain_or_spots",
    1: "needle_line",
    2: "holes",
}


class RoboflowDetector:
    def __init__(
        self,
        api_key: str,
        model_id: str,
        confidence: float,
        api_url: str = ROBOFLOW_API_URL,
        class_remap: dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.confidence = confidence
        self.api_url = api_url.rstrip("/")
        self.remap_enabled = bool(class_remap.get("enabled", False)) if class_remap else False
        self.class_remap = {int(k): int(v) for k, v in class_remap.get("map", {}).items()} if class_remap else {}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            return []
        image_bytes = encoded.tobytes()

        url = f"{self.api_url}/{self.model_id}?api_key={self.api_key}&confidence={self.confidence}"

        req = urllib.request.Request(
            url,
            data=image_bytes,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return []

        predictions = result.get("predictions", [])
        detections: list[Detection] = []
        for pred in predictions:
            cx, cy, bw, bh = pred["x"], pred["y"], pred["width"], pred["height"]
            conf = float(pred["confidence"])
            class_name_raw = str(pred.get("class", ""))

            class_id = ROBOFLOW_CLASS_MAP.get(class_name_raw, 0)
            if self.remap_enabled:
                class_id = self.class_remap.get(class_id, class_id)
            class_name = LOCAL_CLASS_NAMES.get(class_id, f"class_{class_id}")

            x1 = int(max(0, min(w, cx - bw / 2)))
            y1 = int(max(0, min(h, cy - bh / 2)))
            x2 = int(max(0, min(w, cx + bw / 2)))
            y2 = int(max(0, min(h, cy + bh / 2)))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return detections
