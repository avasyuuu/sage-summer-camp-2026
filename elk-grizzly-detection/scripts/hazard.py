"""Gemma-based wildlife hazard assessment from BioCLIP species results."""

import json
import re

DEFAULT_MODEL_REPO = "google/gemma-3-4b-it-qat-q4_0-gguf"
DEFAULT_MODEL_FILE = "gemma-3-4b-it-q4_0.gguf"


class HazardClassifier:
    """Classify an identified species by its inherent potential for harm."""

    def __init__(self, model_repo=DEFAULT_MODEL_REPO,
                 model_file=DEFAULT_MODEL_FILE):
        from llama_cpp import Llama

        self._cache = {}
        self.model = Llama.from_pretrained(
            repo_id=model_repo,
            filename=model_file,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )

    @staticmethod
    def _parse(text):
        """Extract and validate Gemma's JSON response."""
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Gemma did not return JSON: {text!r}")

        result = json.loads(match.group(0))
        hazard = str(result.get("hazard", "")).strip().lower()
        danger_score = result.get("danger_score")
        if hazard not in {"safe", "dangerous"}:
            raise ValueError(f"Invalid hazard label from Gemma: {hazard!r}")
        if isinstance(danger_score, bool) or not isinstance(danger_score, int):
            raise ValueError(
                f"Gemma returned a non-integer danger score: {danger_score!r}"
            )
        if not 1 <= danger_score <= 10:
            raise ValueError(
                f"Gemma returned a danger score outside 1-10: {danger_score!r}"
            )
        if (hazard == "safe" and danger_score > 5) or (
            hazard == "dangerous" and danger_score < 6
        ):
            raise ValueError(
                "Gemma returned an inconsistent hazard label and danger score: "
                f"{hazard!r}, {danger_score!r}"
            )
        return {"hazard": hazard, "danger_score": danger_score}

    def assess(self, common_name, species):
        """Return a species-based hazard label and danger score from 1-10."""
        # Equivalent species always receive the same assessment, regardless of
        # detection confidence or where an image was captured.
        cache_key = (common_name.strip().lower(), species.strip().lower())
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        prompt = f"""Classify an animal's general danger to humans using only its
species identity. Treat the scientific name as the primary identifier.

Common name: {common_name or "unknown"}
Scientific name: {species or "unknown"}

Base the decision only on established characteristics of the identified species.
Do not consider or mention location, surroundings, proximity, current behavior,
detection confidence, or other circumstances.

Assign a danger score from 1 to 10 based on the species' inherent capacity to
harm humans through its typical size, strength, defensive or predatory behavior,
venom, toxins, or well-established disease risk. A score of 1 means the least
dangerous and 10 means the most dangerous. Use "safe" for scores 1-5 and
"dangerous" for scores 6-10. This is a general species classification, not an
assessment of the immediate risk from one particular animal.

Return only JSON in exactly this shape:
{{"hazard":"safe|dangerous","danger_score":1}}"""

        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.0,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "hazard": {
                            "type": "string",
                            "enum": ["safe", "dangerous"],
                        },
                        "danger_score": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["hazard", "danger_score"],
                    "additionalProperties": False,
                },
            },
        )
        text = response["choices"][0]["message"]["content"] or ""
        result = self._parse(text)
        self._cache[cache_key] = result
        return result.copy()

    def assess_detections(self, detections):
        """Add hazard fields to each detection that has a BioCLIP result."""
        for det in detections:
            if not det.get("species"):
                continue
            result = self.assess(
                det.get("common_name", ""),
                det["species"],
            )
            det.update(result)
        return detections
