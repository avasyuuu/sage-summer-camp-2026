# Wildlife detection and hazard triage

A wildlife-camera pipeline for the Sage platform. Each image goes through three
stages:

1. **YOLO** (`yolo11l`) detects and localizes animals (bounding boxes).
2. **BioCLIP** identifies the species in each animal crop (scientific + common name).
3. **Gemma 3** classifies the species as `safe` or `dangerous` and assigns a
   danger score from 1 to 10.

Confirmed **dangerous** animals trigger an SMS (Twilio) and a Slack post.

## How it runs today

Sage nodes have limited memory and restricted outbound networking, so **all the
ML and alerting runs on one machine you control** (a DGX, a workstation, a
laptop). Sage is used as the *image source*:

```
Sage beehive  ──pull_images.py──►  query_images/  ──main.py──►  annotated images
(node uploads)                                                  + detections.csv
                                                                + SMS / Slack
```

Everything below happens in one folder, one virtual environment.

## Layout

```
danger-detection/
├── scripts/
│   ├── main.py          entry point — run the pipeline over images
│   ├── pull_images.py   download images from the Sage beehive
│   ├── pipeline.py      orchestrates the three models, annotates, writes the CSV
│   ├── detector.py      YOLO          (finds animals)
│   ├── species.py       BioCLIP       (identifies the species)
│   ├── hazard.py        Gemma 3       (safe / dangerous + score)
│   ├── tracker.py       follows the same animal across frames
│   ├── alerts.py        Twilio SMS + Slack delivery
│   ├── publisher.py     publish to the beehive (only used on a Sage node)
│   └── tests/
├── test_images/         sample stills
├── test_sequences/      ordered frames for tracking tests
├── query_images/        images pulled from the beehive   (git-ignored)
├── output/              annotated images + detections.csv (git-ignored)
├── twilio.env           your credentials                  (git-ignored)
├── requirements.txt
├── Dockerfile           for deploying to a Sage node later
└── sage.yaml            Sage app manifest
```

## Setup

> **Install into the same Python you run the code with.** Most "works on my
> machine" problems are really "the packages are in a different Python." Pick
> one venv and stick with it; use `python -m pip`, not bare `pip`.

```bash
cd danger-detection
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Gemma is gated.** Accept Google's license on the Hugging Face model page, then:

```bash
hf auth login
```

First run downloads ~10 GB of weights (YOLO + BioCLIP + Gemma) into the Hugging
Face cache.

**Credentials** — copy the template and fill it in. `twilio.env` is git-ignored
and must never be committed:

```bash
cp twilio.env.example twilio.env
```

| Variable | Needed for |
|---|---|
| `TWILIO_*`, `ALERT_RECIPIENTS` | SMS alerts |
| `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID` | Slack alerts |
| `SAGE_USER`, `SAGE_USER_TOKEN` | downloading beehive images |

On a Twilio **trial** account, SMS only reaches **verified** numbers.

## Running

**1. Pull images from the beehive**

```bash
python3 scripts/pull_images.py                 # last hour from H00F
python3 scripts/pull_images.py --start -6h --limit 50
python3 scripts/pull_images.py --vsn H03B      # your node, once deployed
```

Querying the beehive is public; *downloading* images needs the Sage token.

**2. Run detection + alerts**

```bash
python3 scripts/main.py query_images     # the images you just pulled
python3 scripts/main.py                  # default: test_images/
python3 scripts/main.py --no-hazard      # skip Gemma (fast iteration)
python3 scripts/main.py --baseline       # first 5 images into output/baseline
```

On start it asks where results go: replace `output/`, add a new subfolder, or
`baseline`. Use `--output replace|new|baseline|<name>` to skip the prompt.

## Outputs

- `output/<name>_detected.jpg` — boxes + species labels, **red** when dangerous
- `output/detections.csv` — species, confidences, hazard verdict, danger score,
  track id, and box coordinates (rewritten each run)

## Alerts

Alerts fire for confirmed **dangerous** tracks. If credentials are missing or
invalid the pipeline **still runs** — it logs `Twilio message failed` /
`Slack message failed` and keeps saving results, so detection never depends on
messaging being configured.

## Deploying to a Sage node (later)

`Dockerfile`, `sage.yaml`, and `scripts/publisher.py` exist for running this
*on* a node, where it would publish detections to the beehive (`--publish`)
instead of alerting directly. That path is not in use yet — the nodes aren't
deployed, and Gemma is large for edge hardware.

## Notes

- **Species scope:** BioCLIP searches the full tree of life. Pass
  `species_labels=[...]` to constrain it and cut low-confidence noise.
- **CPU vs GPU:** runs anywhere; uses CUDA when available. Gemma on CPU is slow.
- **Weights are not committed** (`*.pt` is git-ignored) — they auto-download.

The hazard label describes a species' general capacity to cause harm; it does
not estimate the danger from a particular animal. BioCLIP can misidentify
animals and Gemma can be wrong, so consequential alerts deserve human review.
