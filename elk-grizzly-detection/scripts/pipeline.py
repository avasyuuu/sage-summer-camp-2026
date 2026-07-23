"""End-to-end wildlife pipeline: YOLO detection -> BioCLIP species -> Gemma hazard.

Processes a folder of images and produces, in output/:
  - one annotated image per input (boxes + species label; red box if dangerous)
  - one CSV (detections.csv) with species, confidence, and the Gemma safe/dangerous
    verdict for every detection.
"""

import csv
from pathlib import Path

import cv2

from detector import AnimalDetector
from species import ANIMAL_LABELS, SpeciesClassifier

# Paths resolved from this file, so the pipeline works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "output"
CSV_PATH = OUTPUT_DIR / "detections.csv"
YOLO_WEIGHTS = PROJECT_ROOT / "yolo11n.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CSV_FIELDS = [
    "image",
    "detected_as",       # YOLO's coarse COCO label
    "yolo_confidence",
    "species",           # BioCLIP scientific name
    "common_name",
    "species_confidence",  # BioCLIP "accuracy"
    "hazard",            # Gemma: safe / dangerous
    "reason",            # Gemma's one-line justification
    "x1", "y1", "x2", "y2",
]

GREEN = (0, 255, 0)
RED = (0, 0, 255)


class WildlifePipeline:
    """Loads the three models once, then runs them over any images you give it."""

    def __init__(self, conf=0.35, use_hazard=True, species_labels=None,
                 gemma_model=None):
        # yolo11n.pt auto-downloads by name if the local weight isn't present yet
        # (e.g. a fresh teammate machine or a fresh container).
        weights = str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else "yolo11n.pt"
        self.detector = AnimalDetector(weights, conf=conf)
        self.species = SpeciesClassifier(labels=species_labels)

        self.hazard = None
        if use_hazard:
            try:
                from hazard import DEFAULT_MODEL, HazardClassifier

                self.hazard = HazardClassifier(gemma_model or DEFAULT_MODEL)
            except Exception as e:
                print(f"[gemma] hazard assessment disabled: {type(e).__name__}: {e}")

        # Where results go. run() can override this per batch.
        self.output_dir = OUTPUT_DIR
        self.csv_path = CSV_PATH

    # ------------------------------------------------------------------ per image

    def process_image(self, image_path):
        """Detect -> identify -> assess -> save annotated image. Returns detections."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        detections = self.detector.detect(image)
        for det in detections:
            # YOLO reports coarse COCO labels; only send actual animals to BioCLIP.
            if det["label"] not in ANIMAL_LABELS:
                continue

            species = self.species.classify(image, det["box"])
            if not species:
                continue
            det["species"] = species["species"]
            det["common_name"] = species["common_name"]
            det["species_confidence"] = species["score"]

            if self.hazard:
                verdict = self.hazard.assess(
                    det.get("common_name", ""),
                    det["species"],
                )
                det["hazard"] = verdict["hazard"]
                det["reason"] = verdict["hazard_reason"]

        self._save_annotated(image, image_path, detections)
        return detections

    # ------------------------------------------------------------------ drawing

    @staticmethod
    def _label_text(det):
        # Prefer BioCLIP's identification: common name, then scientific name.
        # Only fall back to YOLO's coarse label when there's no species at all.
        if det.get("common_name"):
            return f"{det['common_name']} {det['species_confidence']:.2f}"
        if det.get("species"):
            return f"{det['species']} {det['species_confidence']:.2f}"
        return f"{det['label']} {det['confidence']:.2f}"

    def _save_annotated(self, image, image_path, detections):
        # Scale line/text weight to the image so labels are legible at any resolution.
        scale = max(image.shape[:2]) / 1000
        thickness = max(int(2 * scale), 2)
        font_scale = max(0.6 * scale, 0.6)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            text = self._label_text(det)
            if det.get("hazard"):
                text += f" [{det['hazard'].upper()}]"
            color = RED if det.get("hazard") == "dangerous" else GREEN

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                image, text,
                (x1, max(y1 - int(8 * scale), int(30 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        src = Path(image_path)
        out_path = self.output_dir / f"{src.stem}_detected{src.suffix}"
        cv2.imwrite(str(out_path), image)
        return out_path

    # ------------------------------------------------------------------ CSV

    @staticmethod
    def _rows_for(image_path, detections):
        name = Path(image_path).name
        if not detections:
            # Record that the image was processed even when nothing was found,
            # so the dataset accounts for every image (useful as a denominator).
            return [{"image": name}]

        rows = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            rows.append(
                {
                    "image": name,
                    "detected_as": det["label"],
                    "yolo_confidence": round(det["confidence"], 4),
                    "species": det.get("species", ""),
                    "common_name": det.get("common_name", ""),
                    "species_confidence": (
                        round(det["species_confidence"], 4)
                        if "species_confidence" in det else ""
                    ),
                    "hazard": det.get("hazard", ""),
                    "reason": det.get("reason", ""),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                }
            )
        return rows

    def _write_csv(self, rows):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Fresh file each run, so the CSV is exactly the current batch (no dupes).
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------ batch

    @staticmethod
    def _collect(inputs):
        """Expand files and folders into a sorted list of image paths."""
        images = []
        for arg in inputs:
            path = Path(arg)
            if path.is_dir():
                images.extend(
                    sorted(p for p in path.iterdir()
                           if p.suffix.lower() in IMAGE_EXTS)
                )
            elif path.suffix.lower() in IMAGE_EXTS and path.exists():
                images.append(path)
            else:
                print(f"skipping (not an image or folder): {arg}")
        return images

    def run(self, inputs=None, output_dir=None):
        """Process files/folders (default: the test_images folder), write one CSV.

        Pass `output_dir` to send this batch's images + CSV somewhere other than
        the default output/ folder (e.g. a sub-folder like output/output1).
        """
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.csv_path = self.output_dir / "detections.csv"

        images = self._collect(inputs or [str(TEST_IMAGES_DIR)])
        if not images:
            print("no images found")
            return

        all_rows = []
        for i, image_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {image_path.name}")
            try:
                detections = self.process_image(image_path)
                all_rows.extend(self._rows_for(image_path, detections))
                for det in detections:
                    self._print_detection(det)
            except Exception as e:
                # One bad frame shouldn't abort the whole batch.
                print(f"  ERROR: {type(e).__name__}: {e}")

        self._write_csv(all_rows)
        print(f"\nDone: {len(images)} images processed")
        print(f"  images -> {self.output_dir}")
        print(f"  CSV    -> {self.csv_path}")

    @staticmethod
    def _print_detection(det):
        line = f"  {det['label']} {det['confidence']:.2f}"
        if det.get("species"):
            name = det.get("common_name") or det["species"]
            line += f" -> {name} ({det['species']}) {det['species_confidence']:.2f}"
        if det.get("hazard"):
            line += f" -> {det['hazard'].upper()}"
        print(line)
