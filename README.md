# Wildlife detection and hazard triage

A wildlife-camera pipeline for the Sage platform. Each image goes through three
stages:

1. **YOLO** (`yolo11n`) detects and localizes animals (bounding boxes).
2. **BioCLIP** identifies the species in each animal crop (scientific + common name).
3. **Gemma 3** classifies the BioCLIP result as `safe` or `dangerous` using
   only the identified species and gives a short reason.

Outputs, in `output/`:

- one annotated image per input (`<name>_detected.jpg`) — box + species label,
  drawn **red** when Gemma flags it dangerous, green otherwise;
- one CSV, `output/detections.csv`, with the species, confidence, and hazard
  verdict for every detection.

## Setup

> **Install the packages into the same Python you run the code with.** Most
> "works on my machine" problems are really "the packages are in a different
> Python than the one running." Pick one interpreter (a venv or conda env) and
> stick with it. Use `python -m pip` (not bare `pip`) so the install lands in
> that interpreter.

### Option A — local (virtual environment)

```bash
# from the elk-grizzly-detection/ folder
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Gemma access:** `google/gemma-3-4b-it` is gated. Before the first run, accept
Google's Gemma license on the Hugging Face model page and authenticate:

```bash
hf auth login
```

The first run downloads all model weights (YOLO, BioCLIP, and Gemma). Gemma 3
4B is substantially larger than the previous 1B model, so allow extra download
time, disk space, and memory.

### Option B — Docker (matches the Sage deployment)

The container bundles Python + every dependency, so it runs identically on any
machine regardless of local Python setup.

```bash
docker build -t elk-detect .
docker run --rm -v "$(pwd)/output:/app/output" elk-detect
```

The `-v` mount brings the results back out to your `output/` folder.

## Running

```bash
# process every image in test_images/ (the default)
python scripts/main.py

# process specific images
python scripts/main.py test_images/d37363s15i5.jpg test_images/d70380s20i3.jpg

# skip the (slow) Gemma step — hazard columns left blank
python scripts/main.py --no-hazard
```

`detections.csv` is rewritten fresh each run, so it always reflects exactly the
images from that run. Set `GEMMA_MODEL` (or `--gemma-model`) to point at a
different or local Gemma model.

## Layout

```
scripts/
  main.py       entry point (CLI)
  pipeline.py   WildlifePipeline — orchestrates the three models, annotates, writes the CSV
  detector.py   AnimalDetector    — YOLO
  species.py    SpeciesClassifier — BioCLIP
  hazard.py     HazardClassifier  — Gemma 3
test_images/    input images
output/         annotated images + detections.csv  (git-ignored)
```

## Notes

- **Species scope:** BioCLIP classifies against the full tree of life by default.
  To constrain it to your target species, build the pipeline with
  `species_labels=[...]` (see `SpeciesClassifier`).
- **CPU vs GPU:** runs on CPU anywhere; uses CUDA automatically when available.
  Gemma on CPU is the slow part — use `--no-hazard` while iterating.
- **Weights are not committed** (`*.pt` is git-ignored). YOLO auto-downloads;
  BioCLIP/Gemma download from Hugging Face on first use.

The hazard label describes a species' general capacity to cause serious harm;
it does not estimate the immediate danger posed by a particular animal. BioCLIP
can misidentify animals and Gemma can produce incorrect judgments, so
consequential alerts should be reviewed by a person and validated against local
wildlife guidance.
