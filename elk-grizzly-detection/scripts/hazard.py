"""Gemma-based wildlife hazard assessment from BioCLIP species results."""

import json
import re


DEFAULT_MODEL = "google/gemma-3-1b-it"


class HazardClassifier:
    """Classify an identified species as safe or dangerous for a given context."""

    def __init__(self, model_id=DEFAULT_MODEL, context=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.context = context or (
            "The animal was detected by a fixed outdoor camera near people, "
            "homes, trails, or other infrastructure."
        )
        self._cache = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
        )

    @staticmethod
    def _parse(text):
        """Extract and validate Gemma's JSON response."""
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Gemma did not return JSON: {text!r}")

        result = json.loads(match.group(0))
        hazard = str(result.get("hazard", "")).strip().lower()
        reason = str(result.get("reason", "")).strip()
        if hazard not in {"safe", "dangerous"}:
            raise ValueError(f"Invalid hazard label from Gemma: {hazard!r}")
        if not reason:
            raise ValueError("Gemma returned no reason")
        return {"hazard": hazard, "hazard_reason": reason}

    def assess(self, common_name, species, species_score):
        """Return a validated hazard label and short explanation."""
        # Avoid repeating generation for equivalent detections in YOLO and SAM.
        cache_key = (common_name, species, round(species_score, 2), self.context)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        prompt = f"""You are the final stage of a wildlife camera alert system.
YOLO localized an animal and BioCLIP identified it. Decide whether this
detection warrants a safety alert in the stated setting.

Setting: {self.context}
Common name: {common_name or "unknown"}
Scientific name: {species or "unknown"}
BioCLIP confidence: {species_score:.3f}

Use "dangerous" when the identified animal could plausibly threaten people,
pets, livestock, or property in this setting, or when low identification
confidence makes dismissing the detection unsafe. Otherwise use "safe".
This is a triage decision, not a claim that the species is always harmful.

Return only JSON in exactly this shape, with a concise reason:
{{"hazard":"safe|dangerous","reason":"one short sentence"}}"""

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        result = self._parse(self.tokenizer.decode(generated, skip_special_tokens=True))
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
                det["species_score"],
            )
            det.update(result)
        return detections
