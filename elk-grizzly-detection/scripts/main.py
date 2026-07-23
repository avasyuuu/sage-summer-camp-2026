import argparse
import os

import log
from config import model_file
from detector import AnimalDetector
from hazard import DEFAULT_MODEL, HazardClassifier
from species import SpeciesClassifier


def report(name, detector, hazard_classifier, image_path, suffix):
    """Run one pipeline over one image: print, log to CSV, save annotated image."""
    print(f"[{name}]")

    # detect once and reuse, so BioCLIP doesn't run twice per image
    detections = detector.detect(image_path)
    if hazard_classifier:
        hazard_classifier.assess_detections(detections)

    for det in detections:
        line = f"  {det['label']}: {det['confidence']:.2f} at {det['box']}"
        if det.get("species"):
            line += f" -> {det['common_name']} ({det['species']}) {det['species_score']:.2f}"
        if det.get("hazard"):
            line += f" -> {det['hazard'].upper()}: {det['hazard_reason']}"
        print(line)

    csv_path = log.append(detections, image_path, pipeline=name)
    out = detector.annotate(image_path, suffix=suffix, detections=detections)
    print(f"  saved -> {out}, logged -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect, identify, and assess wildlife")
    parser.add_argument("images", nargs="+", help="image paths to process")
    parser.add_argument(
        "--gemma-model",
        default=os.environ.get("GEMMA_MODEL", DEFAULT_MODEL),
        help="Hugging Face Gemma model ID or local model directory",
    )
    parser.add_argument(
        "--context",
        default=os.environ.get("HAZARD_CONTEXT"),
        help="location context Gemma should use for the safety decision",
    )
    parser.add_argument(
        "--no-hazard",
        action="store_true",
        help="run only YOLO and BioCLIP",
    )
    args = parser.parse_args()

    # One classifier shared by both pipelines so BioCLIP is loaded once.
    # labels=None classifies against the full tree of life; pass a list to
    # constrain it, e.g. ["Cervus canadensis", "Ursus arctos horribilis"].
    classifier = SpeciesClassifier()

    hazard_classifier = None
    if not args.no_hazard:
        try:
            hazard_classifier = HazardClassifier(args.gemma_model, context=args.context)
        except Exception as e:
            print(f"[gemma] unavailable, hazard assessment disabled: {type(e).__name__}: {e}")

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

    for image_path in args.images:
        print(f"\n=== {image_path} ===")
        report("yolo", yolo, hazard_classifier, image_path, suffix="yolo")
        if sam:
            report("sam", sam, hazard_classifier, image_path, suffix="sam")


if __name__ == "__main__":
    main()
