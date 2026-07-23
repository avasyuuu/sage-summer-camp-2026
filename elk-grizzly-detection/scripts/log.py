"""Append detections to a CSV for later analysis."""

import csv
from datetime import datetime, timezone
from pathlib import Path

from config import output_file

FIELDS = [
    "timestamp",
    "image",
    "pipeline",
    "label",
    "confidence",
    "species",
    "common_name",
    "species_score",
    "hazard",
    "hazard_reason",
    "x1",
    "y1",
    "x2",
    "y2",
]

GEMMA_FIELDS = [
    "timestamp",
    "image",
    "pipeline",
    "yolo_label",
    "yolo_confidence",
    "species",
    "common_name",
    "bioclip_confidence",
    "prediction",
    "reason",
]


def append(detections, image_path, pipeline, csv_path=None):
    """Append one row per detection. Creates the file with a header if needed.

    Defaults to output/detections.csv. Opens in append mode so results
    accumulate across runs; delete the file to start a fresh dataset.
    """
    path = Path(csv_path) if csv_path else output_file("detections.csv")
    new_file = not path.exists() or path.stat().st_size == 0
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "image": Path(image_path).name,
                    "pipeline": pipeline,
                    "label": det["label"],
                    "confidence": round(det["confidence"], 4),
                    "species": det.get("species", ""),
                    "common_name": det.get("common_name", ""),
                    "species_score": (
                        round(det["species_score"], 4)
                        if det.get("species_score") is not None
                        else ""
                    ),
                    "hazard": det.get("hazard", ""),
                    "hazard_reason": det.get("hazard_reason", ""),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return path


def append_gemma(detections, image_path, pipeline, csv_path=None):
    """Append Gemma safety predictions to a dedicated, concise CSV."""
    path = Path(csv_path) if csv_path else output_file("gemma_predictions.csv")
    new_file = not path.exists() or path.stat().st_size == 0
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GEMMA_FIELDS)
        if new_file:
            writer.writeheader()

        for det in detections:
            if not det.get("hazard"):
                continue
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "image": Path(image_path).name,
                    "pipeline": pipeline,
                    "yolo_label": det["label"],
                    "yolo_confidence": round(det["confidence"], 4),
                    "species": det.get("species", ""),
                    "common_name": det.get("common_name", ""),
                    "bioclip_confidence": round(det["species_score"], 4),
                    "prediction": det["hazard"],
                    "reason": det.get("hazard_reason", ""),
                }
            )

    return path
