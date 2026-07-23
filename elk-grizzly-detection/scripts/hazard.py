"""Gemma-based wildlife hazard assessment from BioCLIP species results."""

import json
import os
import re

# Gemma 3's generate() path tries to torch.compile the model. The compiler's
# backend (Inductor) needs Triton to generate kernels, and Triton has no
# Windows support, so the compile step raises TritonMissing. Force plain eager
# execution instead — no compile, no Triton. This must be set before torch's
# dynamo module is first imported, so it lives at module top. Eager is more
# than fast enough for one detection at a time.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


DEFAULT_MODEL = "google/gemma-3-4b-it"


class HazardClassifier:
    """Classify an identified species by its inherent potential for harm."""

    def __init__(self, model_id=DEFAULT_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Belt-and-suspenders with the env var above: disable the compiler at
        # runtime too, in case torch's dynamo module was already imported.
        try:
            torch._dynamo.config.disable = True
        except Exception:
            pass

        self._cache = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
        )

        # We decode greedily (do_sample=False), so the model's default sampling
        # settings never apply. Clear them so transformers stops warning that
        # top_p/top_k are ignored. This doesn't change any output.
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

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

    def assess(self, common_name, species):
        """Return a species-based hazard label and short explanation."""
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

Use "dangerous" when members of the species have a meaningful inherent capacity
to cause serious injury or death to humans through their typical size, strength,
defensive or predatory behavior, venom, toxins, or well-established disease
risk. Otherwise use "safe". This is a general species classification, not an
assessment of the immediate risk from one particular animal.

Return only JSON in exactly this shape, with a concise species-based reason:
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
            )
            det.update(result)
        return detections
