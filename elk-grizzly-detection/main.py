# from waggle.plugin import Plugin
# from waggle.data.vision import Camera
import sys

from detector import AnimalDetector
from species import SpeciesClassifier


def main():
    if len(sys.argv) < 2:
        print("usage: python main.py <image> [image ...]")
        return

    # labels=None classifies against the full tree of life.
    # Pass a list to constrain it, e.g.:
    #   SpeciesClassifier(labels=["Cervus canadensis", "Ursus arctos horribilis"])
    detector = AnimalDetector(
        model_path="yolo11n.pt",
        conf=0.35,
        species_classifier=SpeciesClassifier(),
    )

    for image_path in sys.argv[1:]:
        for det in detector.detect(image_path):
            line = f"  {det['label']}: {det['confidence']:.2f} at {det['box']}"
            if det.get("species"):
                line += f" -> {det['common_name']} ({det['species']}) {det['species_score']:.2f}"
            print(line)

        out = detector.annotate(image_path)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
