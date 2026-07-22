"""Minimal YOLO wildlife detector, with optional BioCLIP species ID."""

from pathlib import Path

import cv2
from ultralytics import YOLO

from species import ANIMAL_LABELS


class AnimalDetector:
    """Wraps a YOLO model: image in, annotated image out.

    Pass a `SpeciesClassifier` as `species_classifier` to run BioCLIP on each
    animal crop, which turns YOLO's coarse COCO label ('bear', 'bird') into
    an actual species.
    """

    def __init__(self, model_path="yolo11n.pt", conf=0.35, classes=None,
                 species_classifier=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes  # e.g. ["bear", "horse"] to filter, None = keep all
        self.species_classifier = species_classifier

    def _read(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image

    def detect(self, image_path, image=None):
        """Run the model and return a list of detection dicts.

        Each dict has label/confidence/box, plus species/common_name/species_score
        when a species classifier is attached and the detection is an animal.
        """
        if image is None:
            image = self._read(image_path)

        result = self.model(image, conf=self.conf, verbose=False)[0]

        detections = []
        for box in result.boxes:
            label = result.names[int(box.cls)]
            if self.classes and label not in self.classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det = {
                "label": label,
                "confidence": float(box.conf),
                "box": (int(x1), int(y1), int(x2), int(y2)),
            }

            if self.species_classifier and label in ANIMAL_LABELS:
                species = self.species_classifier.classify(image, det["box"])
                if species:
                    det["species"] = species["species"]
                    det["common_name"] = species["common_name"]
                    det["species_score"] = species["score"]

            detections.append(det)
        return detections

    def annotate(self, image_path, output_path=None):
        """Draw boxes + labels on the image and save it. Returns the output path."""
        image = self._read(image_path)

        # scale line/text weight to the image so labels stay readable at any resolution
        scale = max(image.shape[:2]) / 1000
        thickness = max(int(2 * scale), 2)
        font_scale = max(0.6 * scale, 0.6)

        for det in self.detect(image_path, image=image):
            x1, y1, x2, y2 = det["box"]

            # prefer the species name when BioCLIP identified one
            if det.get("common_name"):
                text = f"{det['common_name']} {det['species_score']:.2f}"
            else:
                text = f"{det['label']} {det['confidence']:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(
                image,
                text,
                (x1, max(y1 - int(8 * scale), int(30 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness,
            )

        if output_path is None:
            src = Path(image_path)
            output_path = src.with_name(f"{src.stem}_detected{src.suffix}")

        cv2.imwrite(str(output_path), image)
        return output_path
