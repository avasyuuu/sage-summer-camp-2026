# from waggle.plugin import Plugin
# from waggle.data.vision import Camera
import sys

import log
from config import model_file
from detector import AnimalDetector
from species import SpeciesClassifier


def report(name, detector, image_path, suffix):
    """Run one pipeline over one image: print, log to CSV, save annotated image."""
    print(f"[{name}]")

    # detect once and reuse, so BioCLIP doesn't run twice per image
    detections = detector.detect(image_path)

    for det in detections:
        line = f"  {det['label']}: {det['confidence']:.2f} at {det['box']}"
        if det.get("species"):
            line += f" -> {det['common_name']} ({det['species']}) {det['species_score']:.2f}"
        print(line)

    csv_path = log.append(detections, image_path, pipeline=name)
    out = detector.annotate(image_path, suffix=suffix, detections=detections)
    print(f"  saved -> {out}, logged -> {csv_path}")


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/main.py <image> [image ...]")
        return

    # One classifier shared by both pipelines so BioCLIP is loaded once.
    # labels=None classifies against the full tree of life; pass a list to
    # constrain it, e.g. ["Cervus canadensis", "Ursus arctos horribilis"].
    classifier = SpeciesClassifier()

    yolo = AnimalDetector(
        model_path=str(model_file("yolo11n.pt")),
        conf=0.35,
        species_classifier=classifier,
    )

    # SAM 3 weights are gated and must be downloaded manually; keep the YOLO
    # pipeline usable when they're absent.
    sam = None
    try:
        from sam_detector import SamDetector

        sam = SamDetector(
            model_path=str(model_file("sam3.pt")),
            prompts=["elk", "grizzly bear", "deer", "black bear"],
            species_classifier=classifier,
        )
    except Exception as e:
        print(f"[sam] unavailable, skipping: {type(e).__name__}: {e}")

    for image_path in sys.argv[1:]:
        print(f"\n=== {image_path} ===")
        report("yolo", yolo, image_path, suffix="yolo")
        if sam:
            report("sam", sam, image_path, suffix="sam")


if __name__ == "__main__":
    main()
