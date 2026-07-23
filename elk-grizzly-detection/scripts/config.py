"""Project paths, resolved relative to this file so scripts work from any cwd."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Model weights live at the project root (yolo11n.pt auto-downloads there;
# sam3.pt must be placed there manually).
MODELS_DIR = PROJECT_ROOT


def output_file(name):
    """Return a path inside output/, creating the directory if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def model_file(name):
    """Return the full path to a weights file."""
    return MODELS_DIR / name
