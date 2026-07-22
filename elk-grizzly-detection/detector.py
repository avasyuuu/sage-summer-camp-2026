"""YOLO detection pipeline, with optional BioCLIP species ID."""

from ultralytics import YOLO

from common import BaseDetector, trusted_checkpoint_load


class AnimalDetector(BaseDetector):
    """Wraps a YOLO model: image in, annotated image out.

    Pass a `SpeciesClassifier` as `species_classifier` to run BioCLIP on each
    animal crop, which turns YOLO's coarse COCO label ('bear', 'bird') into
    an actual species.
    """

    def __init__(self, model_path="yolo11n.pt", conf=0.35, classes=None,
                 species_classifier=None):
        super().__init__(species_classifier=species_classifier)
        with trusted_checkpoint_load():
            self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes  # e.g. ["bear", "horse"] to filter, None = keep all

    def _predict(self, image):
        result = self.model(image, conf=self.conf, verbose=False)[0]

        detections = []
        for box in result.boxes:
            label = result.names[int(box.cls)]
            if self.classes and label not in self.classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "label": label,
                    "confidence": float(box.conf),
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                }
            )
        return detections
