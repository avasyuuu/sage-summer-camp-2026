"""Run the wildlife pipeline (YOLO -> BioCLIP -> Gemma) over images.

Beehive publishing lives in `publisher.py`, not here: `from waggle.plugin
import Plugin` and the `plugin.publish()` / `plugin.upload_file()` calls are
inside `open_publisher()` and `BeehivePublisher`. They are imported lazily so
this runs on machines with no Waggle runtime; pass --publish to enable them.

Not implemented yet: `waggle.data.vision.Camera`. Images currently come from a
folder (see `pull_images.py`), not from a node's camera.
"""

import argparse
import os
import sys
from pathlib import Path

from alerts import AlertConfigurationError, load_alert_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "output"


def _next_subfolder(base):
    """Return the next free output/outputN folder (output1, output2, ...)."""
    n = 1
    while (base / f"output{n}").exists():
        n += 1
    return base / f"output{n}"


def _clear_top_level(base):
    """Delete the files directly in `base` (leaves saved sub-folders intact)."""
    if base.exists():
        for p in base.iterdir():
            if p.is_file():
                p.unlink()


BASELINE_LIMIT = 5  # baseline runs only the first N images


def choose_output_dir(explicit):
    """Decide where this run's results go; returns (output_dir, limit).

    `limit` is None except for the baseline option, which processes only the
    first few images. `explicit` is the --output / --baseline value (or None).
    With no explicit choice and an interactive terminal, ask the user; with no
    terminal (e.g. a container), default to replacing the current output so
    nothing hangs.
    """
    if explicit:
        choice = explicit.strip()
    elif sys.stdin.isatty():
        print("Where should the results go?")
        print("  [1] Replace the current output   (output/)")
        print("  [2] Add new output               (output/output1, output2, ...)")
        print(f"  [3] Baseline                     (first {BASELINE_LIMIT} images -> output/baseline)")
        choice = input("Choice [1/2/3, default 1]: ").strip() or "1"
    else:
        choice = "1"  # no interactive terminal: don't block, just replace

    low = choice.lower()
    if low in ("", "1", "replace"):
        _clear_top_level(OUTPUT_DIR)
        print(f"-> replacing current output: {OUTPUT_DIR}\n")
        return OUTPUT_DIR, None
    if low in ("2", "new"):
        target = _next_subfolder(OUTPUT_DIR)
        print(f"-> new output folder: {target}\n")
        return target, None
    if low in ("3", "baseline"):
        target = OUTPUT_DIR / "baseline"
        _clear_top_level(target)  # baseline folder is replaced each time
        print(f"-> baseline, first {BASELINE_LIMIT} images: {target}\n")
        return target, BASELINE_LIMIT
    # anything else is treated as a custom sub-folder name under output/
    target = OUTPUT_DIR / choice
    print(f"-> sub-folder: {target}\n")
    return target, None


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
        "--min-species-confidence",
        type=float,
        default=None,
        help="ignore BioCLIP identifications below this confidence "
             "(default: 0.3)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish dangerous detections + annotated images to the Sage "
             "beehive (for an off-node watcher to alert from)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="baseline run: first 5 images into output/baseline (replaced each time)",
    )
    parser.add_argument(
        "--output",
        help="skip the prompt: 'replace', 'new', 'baseline', or a custom folder name",
    )
    parser.add_argument(
        "--gemma-model-id",
        default=os.environ.get("GEMMA_MODEL_ID"),
        help="Hugging Face model ID for Gemma (default: google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--confirmation-frames",
        type=int,
        default=1,
        metavar="FRAMES",
        help="consecutive sightings required to confirm an animal (default: 1)",
    )
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=30,
        metavar="FRAMES",
        help="consecutive misses before forgetting an animal track (default: 30)",
    )
    args = parser.parse_args()

    if args.confirmation_frames < 1:
        parser.error("--confirmation-frames must be at least 1")
    if args.max_missed_frames < 1:
        parser.error("--max-missed-frames must be at least 1")

    try:
        alert_config = load_alert_config(PROJECT_ROOT / "twilio.env")
    except AlertConfigurationError as exc:
        # Don't kill the run: a node without credentials should still detect,
        # annotate, and log. Alerts report themselves as failed instead.
        print(f"[alerts] configuration unavailable: {exc}")
        print("[alerts] running in detection-only mode - Twilio and Slack "
              "messages will NOT be sent.")
        alert_config = None

    # The --baseline flag is the same as choosing baseline at the prompt.
    explicit = "baseline" if args.baseline else args.output

    # Ask where results should go before loading the (slow) models.
    output_dir, limit = choose_output_dir(explicit)

    # Import the ML pipeline only after alert configuration has passed startup
    # validation, so a broken deployment fails before loading any model.
    from pipeline import WildlifePipeline
    from publisher import open_publisher

    publisher, plugin = open_publisher(args.publish)
    try:
        pipeline = WildlifePipeline(
            use_hazard=not args.no_hazard,
            gemma_model_id=args.gemma_model_id,
            alert_config=alert_config,
            confirmation_frames=args.confirmation_frames,
            max_missed_frames=args.max_missed_frames,
            publisher=publisher,
            **({"min_species_confidence": args.min_species_confidence}
               if args.min_species_confidence is not None else {}),
        )
        pipeline.run(args.images, output_dir=output_dir, limit=limit)
    finally:
        if plugin is not None:
            plugin.__exit__(None, None, None)


if __name__ == "__main__":
    main()
