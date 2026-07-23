# Wildlife detection and hazard triage

The pipeline runs three stages:

1. YOLO detects and localizes animals.
2. BioCLIP identifies the species in each animal crop.
3. Gemma classifies the BioCLIP result as `safe` or `dangerous` for the stated
   location context and gives a short reason.

## Gemma setup

The default model is `google/gemma-3-1b-it`. Before the first run, sign in to
Hugging Face, accept Google's Gemma license on the model page, and authenticate:

```bash
hf auth login
pip install -r requirements.txt
```

The first run downloads the model weights. To use a previously downloaded local
model, set `GEMMA_MODEL` to its directory.

## Run

```bash
python scripts/main.py example.jpg
```

Provide the real deployment setting so Gemma evaluates risk in context:

```bash
python scripts/main.py \
  --context "Camera beside a campground in northern Wisconsin" \
  example.jpg
```

Use `--no-hazard` to run only YOLO and BioCLIP. Results are written to
`output/detections.csv`; annotated boxes are red for `dangerous` and green for
`safe`.

The hazard label is triage guidance, not a guarantee of safety. BioCLIP can
misidentify animals and Gemma can produce incorrect judgments, so consequential
alerts should be reviewed by a person and validated against local wildlife rules.
