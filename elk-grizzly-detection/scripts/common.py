"""Shared plumbing for the YOLO and SAM detection pipelines."""

import os
from contextlib import contextmanager
from pathlib import Path

import cv2

from config import output_file
from species import ANIMAL_LABELS


@contextmanager
def trusted_checkpoint_load():
    """Let ultralytics load its own checkpoint.

    Importing bioclip sets TORCH_FORCE_WEIGHTS_ONLY_LOAD=true, which torch
    re-reads on every torch.load() and which overrides the weights_only=False
    that ultralytics needs. We relax it only around the model load, so this
    applies to the official YOLO/SAM weights and nothing else. Don't widen
    this to checkpoints you didn't get from a trusted source.
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


class BaseDetector:
    """Common behaviour: read, run species ID, draw boxes, save.

    Subclasses implement `_predict(image)` and return a list of
    {label, confidence, box} dicts. Everything downstream is shared so the
    two pipelines annotate identically and stay comparable.
    """

    # YOLO reports coarse COCO labels, so only animal crops are worth sending
    # to BioCLIP. SAM's labels are the concepts you asked for, so it skips
    # this gate and classifies everything it returns.
    gate_labels = True

    def __init__(self, species_classifier=None):
        self.species_classifier = species_classifier

    def _predict(self, image):
        raise NotImplementedError

    def _read(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image

    def detect(self, image_path, image=None):
        """Return detection dicts, enriched with species when a classifier is set."""
        if image is None:
            image = self._read(image_path)

        detections = self._predict(image)

        if self.species_classifier:
            for det in detections:
                if self.gate_labels and det["label"] not in ANIMAL_LABELS:
                    continue
                species = self.species_classifier.classify(image, det["box"])
                if species:
                    det["species"] = species["species"]
                    det["common_name"] = species["common_name"]
                    det["species_score"] = species["score"]

        return detections

    def annotate(self, image_path, output_path=None, suffix="detected",
                 detections=None):
        """Draw boxes + labels on the image and save it. Returns the output path.

        Pass `detections` from an earlier detect() call to avoid running
        inference (and BioCLIP) a second time.
        """
        image = self._read(image_path)

        # scale line/text weight to the image so labels stay readable at any resolution
        scale = max(image.shape[:2]) / 1000
        thickness = max(int(2 * scale), 2)
        font_scale = max(0.6 * scale, 0.6)

        if detections is None:
            detections = self.detect(image_path, image=image)

        for det in detections:
            x1, y1, x2, y2 = det["box"]

            # prefer the species name when BioCLIP identified one
            if det.get("common_name"):
                text = f"{det['common_name']} {det['species_score']:.2f}"
            else:
                text = f"{det['label']} {det['confidence']:.2f}"

            if det.get("hazard"):
                text += f" {det['hazard'].upper()}"

            color = (0, 0, 255) if det.get("hazard") == "dangerous" else (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                image,
                text,
                (x1, max(y1 - int(8 * scale), int(30 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
            )

        if output_path is None:
            src = Path(image_path)
            output_path = output_file(f"{src.stem}_{suffix}{src.suffix}")

        cv2.imwrite(str(output_path), image)
        return output_path
