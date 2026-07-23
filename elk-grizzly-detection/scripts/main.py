# from waggle.plugin import Plugin
# from waggle.data.vision import Camera
"""Run the wildlife pipeline (YOLO -> BioCLIP -> Gemma) over images."""

import argparse
import os

from pipeline import TEST_IMAGES_DIR, WildlifePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Detect animals (YOLO), identify species (BioCLIP), and "
                    "assess hazard (Gemma). Saves annotated images + one CSV."
    )
    parser.add_argument(
        "images",
        nargs="*",
        default=[str(TEST_IMAGES_DIR)],
        help="image files or folders to process (default: the test_images folder)",
    )
    parser.add_argument(
        "--no-hazard",
        action="store_true",
        help="skip the Gemma hazard step (faster; hazard columns left blank)",
    )
    parser.add_argument(
        "--context",
        default=os.environ.get("HAZARD_CONTEXT"),
        help="location context Gemma uses for the safety decision",
    )
    parser.add_argument(
        "--gemma-model",
        default=os.environ.get("GEMMA_MODEL"),
        help="Hugging Face Gemma model id or local model directory",
    )
    args = parser.parse_args()

    pipeline = WildlifePipeline(
        use_hazard=not args.no_hazard,
        gemma_model=args.gemma_model,
        context=args.context,
    )
    pipeline.run(args.images)


if __name__ == "__main__":
    main()
