"""Unit tests for the Transformers-backed Gemma hazard classifier."""

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from hazard import HazardClassifier  # noqa: E402


class FakePipeline:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class HazardTransformersTests(unittest.TestCase):
    def test_assess_uses_transformers_chat_and_caches_species(self):
        inference = FakePipeline(
            [[{"generated_text": '{"hazard":"dangerous","danger_score":8}'}]]
        )
        classifier = HazardClassifier(inference_pipeline=inference)

        first = classifier.assess("Leopard", "Panthera pardus")
        second = classifier.assess("Leopard", "Panthera pardus")

        self.assertEqual(first, {"hazard": "dangerous", "danger_score": 8})
        self.assertEqual(second, first)
        self.assertEqual(len(inference.calls), 1)
        call = inference.calls[0]
        self.assertFalse(call["return_full_text"])
        self.assertEqual(call["generate_kwargs"]["max_new_tokens"], 80)
        self.assertFalse(call["generate_kwargs"]["do_sample"])
        self.assertEqual(call["text"][0]["role"], "user")
        self.assertIn("Panthera pardus", call["text"][0]["content"][0]["text"])

    def test_chat_message_response_is_supported(self):
        response = [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": '{"hazard":"safe","danger_score":3}',
                    }
                ]
            }
        ]
        classifier = HazardClassifier(inference_pipeline=FakePipeline([response]))

        self.assertEqual(
            classifier.assess("Elk", "Cervus canadensis"),
            {"hazard": "safe", "danger_score": 3},
        )

    def test_invalid_transformers_response_is_rejected(self):
        classifier = HazardClassifier(inference_pipeline=FakePipeline([{"text": ""}]))

        with self.assertRaisesRegex(ValueError, "invalid response"):
            classifier.assess("Elk", "Cervus canadensis")


if __name__ == "__main__":
    unittest.main()
