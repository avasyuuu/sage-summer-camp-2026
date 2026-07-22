"""BioCLIP species classification for cropped animal detections."""

import cv2
from PIL import Image

# COCO classes YOLO can report that are worth sending to BioCLIP.
# Elk usually land on 'horse' or 'cow'; grizzlies on 'bear'.
ANIMAL_LABELS = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}


class SpeciesClassifier:
    """Identifies the species inside a cropped detection.

    Pass `labels` to constrain the model to a known set of candidates
    (more reliable when you only care about a few species); leave it None
    to classify against the full tree of life.
    """

    def __init__(self, labels=None, padding=0.05):
        if labels:
            from bioclip import CustomLabelsClassifier

            self.classifier = CustomLabelsClassifier(labels)
        else:
            from bioclip import Rank, TreeOfLifeClassifier

            self.classifier = TreeOfLifeClassifier()
            self.rank = Rank.SPECIES

        self.labels = labels
        self.padding = padding  # fraction of box size to include around the crop

    def _crop(self, image, box):
        """Crop a box out of a BGR image, with padding, as a PIL RGB image."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box

        pad_x = int((x2 - x1) * self.padding)
        pad_y = int((y2 - y1) * self.padding)

        x1 = max(x1 - pad_x, 0)
        y1 = max(y1 - pad_y, 0)
        x2 = min(x2 + pad_x, w)
        y2 = min(y2 + pad_y, h)

        crop = image[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def classify(self, image, box):
        """Return {species, common_name, score} for one detection, or None."""
        crop = self._crop(image, box)
        if self.labels:
            results = self.classifier.predict([crop])
        else:
            results = self.classifier.predict([crop], rank=self.rank, k=1)
        if not results:
            return None

        top = results[0]
        if self.labels:
            # CustomLabelsClassifier returns {classification, score}
            return {
                "species": top["classification"],
                "common_name": top["classification"],
                "score": float(top["score"]),
            }

        # TreeOfLifeClassifier returns full taxonomy
        return {
            "species": top.get("species", ""),
            "common_name": top.get("common_name", ""),
            "score": float(top["score"]),
        }
