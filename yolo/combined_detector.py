from __future__ import annotations

from typing import Any

import numpy as np

from yolo.detector import Detection


def box_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def merge_detections(
    all_detections: list[Detection],
    iou_threshold: float = 0.5,
    same_class_only: bool = True,
) -> list[Detection]:
    if not all_detections:
        return []

    sorted_dets = sorted(all_detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []

    for det in sorted_dets:
        duplicate = False
        for kept_det in kept:
            if same_class_only and kept_det.class_id != det.class_id:
                continue
            if iou(det.bbox, kept_det.bbox) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)

    return kept


class CombinedDetector:
    def __init__(
        self,
        detectors: list[Any],
        merge_iou_threshold: float = 0.5,
        same_class_only: bool = True,
    ) -> None:
        self.detectors = detectors
        self.merge_iou_threshold = merge_iou_threshold
        self.same_class_only = same_class_only

    def detect(self, frame: np.ndarray) -> list[Detection]:
        all_detections: list[Detection] = []
        for detector in self.detectors:
            try:
                dets = detector.detect(frame)
                all_detections.extend(dets)
            except Exception:
                continue

        return merge_detections(
            all_detections,
            iou_threshold=self.merge_iou_threshold,
            same_class_only=self.same_class_only,
        )
