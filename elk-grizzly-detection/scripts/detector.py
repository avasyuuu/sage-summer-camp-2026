"""YOLO animal detection."""

import os
from contextlib import contextmanager

from ultralytics import YOLO


@contextmanager
def trusted_checkpoint_load():
    """Let ultralytics load its own checkpoint.

    Importing bioclip sets TORCH_FORCE_WEIGHTS_ONLY_LOAD=true, which torch
    re-reads on every torch.load() and which overrides the weights_only=False
    ultralytics needs. Relax it only around the YOLO load, for the official
    weights and nothing else.
    """
    key = "TORCH_FORCE_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(key)
    os.environ[key] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


class AnimalDetector:
    """Wraps a YOLO model: image array in, list of detection dicts out."""

    def __init__(self, model_path="yolo11m.pt", conf=0.35):
        with trusted_checkpoint_load():
            self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image):
        """Return [{label, confidence, box:(x1,y1,x2,y2)}] for a BGR image array."""
        result = self.model(image, conf=self.conf, verbose=False)[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "label": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                }
            )
        return detections
