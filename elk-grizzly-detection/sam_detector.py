"""SAM 3 concept-segmentation pipeline, with optional BioCLIP species ID.

SAM 3 takes text prompts ('elk', 'grizzly bear') and returns every instance of
that concept, so it does the detector's job without COCO's fixed class list.
Weights are gated: request access at https://huggingface.co/facebook/sam3 and
put sam3.pt next to this file. They do NOT auto-download the way YOLO's do.
"""

from pathlib import Path

from common import BaseDetector, trusted_checkpoint_load

DEFAULT_PROMPTS = ["elk", "grizzly bear", "deer", "black bear"]


class SamDetector(BaseDetector):
    """Text-prompted detector built on SAM 3.

    `prompts` are noun phrases describing what to find. Everything SAM returns
    was explicitly asked for, so unlike the YOLO pipeline there's no COCO
    animal gate before BioCLIP.
    """

    gate_labels = False

    def __init__(self, model_path="sam3.pt", prompts=None, conf=0.25,
                 species_classifier=None):
        super().__init__(species_classifier=species_classifier)

        # ultralytics builds the model lazily on first predict, so fail here
        # instead of halfway through a run.
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found. SAM 3 weights are gated and do not "
                "auto-download: request access at "
                "https://huggingface.co/facebook/sam3 and place the file here."
            )

        from ultralytics.models.sam import SAM3SemanticPredictor

        self.prompts = prompts or DEFAULT_PROMPTS
        with trusted_checkpoint_load():
            self.predictor = SAM3SemanticPredictor(
                overrides=dict(
                    conf=conf,
                    task="segment",
                    mode="predict",
                    model=model_path,
                    save=False,
                    verbose=False,
                )
            )

    def _predict(self, image):
        self.predictor.set_image(image)
        result = self.predictor(text=self.prompts)[0]

        detections = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            index = int(box.cls)
            # names maps back to the prompt that matched; fall back to the
            # prompt list if the predictor doesn't populate it
            names = getattr(result, "names", None)
            label = names.get(index) if isinstance(names, dict) else None
            if label is None:
                label = self.prompts[index] if index < len(self.prompts) else str(index)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "label": label,
                    "confidence": float(box.conf),
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                }
            )
        return detections
