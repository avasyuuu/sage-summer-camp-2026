# Danger Detection

Spots wildlife in camera images, identifies the species, decides whether it's
dangerous, and texts/Slacks an alert if it is.

```
image ─► YOLO ─────► BioCLIP ────────► Gemma 3 ──────► Twilio SMS
         finds       identifies        safe or         + Slack post
         animals     the species       dangerous?      (if dangerous)
```

Built for the [Sage](https://sagecontinuum.org) platform. Sage camera nodes
supply the images; the detection runs on a machine you control.

---

## Quick start

**Requirements:** Python 3.10+, ~15 GB free disk (model weights), and a GPU if
you want it fast.

### 1. Clone and enter the project

```bash
git clone https://github.com/avasyuuu/sage-summer-camp-2026.git
cd sage-summer-camp-2026/danger-detection
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
```

> Your prompt should now start with `(.venv)`. Re-run this in every new shell.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify your GPU (optional but recommended)

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

`True` means the models will use the GPU. `False` is fine — everything still
runs on CPU, just slower.

### 5. Log in to Hugging Face

Gemma is a gated model. Accept the license on the
[model page](https://huggingface.co/google/gemma-3-4b-it), then:

```bash
hf auth login
```

### 6. Add your credentials

```bash
cp twilio.env.example twilio.env
```

Edit `twilio.env` and fill in the values:

| Variable | Used for |
|---|---|
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | Twilio account |
| `TWILIO_FROM_NUMBER` | the number alerts are sent *from* |
| `ALERT_RECIPIENTS` | who gets the SMS (comma-separated, E.164) |
| `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID` | Slack alerts |
| `SAGE_USER`, `SAGE_USER_TOKEN` | downloading images from Sage |

`twilio.env` is git-ignored — never commit it.

> **Twilio trial accounts can only text verified numbers.** Add each recipient
> under *Verified Caller IDs* in the Twilio console first.

### 7. Run it

```bash
python3 scripts/main.py --baseline --no-hazard
```

A fast check with no alerts. If you see detections, you're set up correctly.

---

## How it works

Each image passes through four stages:

| Stage | Model | What it does |
|---|---|---|
| **Detect** | YOLO11-large | Finds animals and draws boxes. Only knows coarse classes (`bear`, `dog`, `bird`), so its label is often wrong. |
| **Identify** | BioCLIP 2 | Takes each cropped box and names the actual species from the biological tree of life. This is the label to trust. |
| **Assess** | Gemma 3 4B | Reads the species name and returns `safe`/`dangerous` plus a 1–10 danger score. |
| **Alert** | Twilio + Slack | Sends one alert per dangerous animal. |

**A tracker sits across frames** so the same animal keeps one identity and
alerts only once, no matter how long it stays in view. A track is dropped
after 30 frames without a match; if the animal returns, it counts as new.

**Identifications below 0.3 confidence are ignored** — BioCLIP always returns
its best guess, and below that threshold it's guessing.

> YOLO calling a raccoon a `cat` is normal and expected. That's exactly why
> BioCLIP runs afterward. Trust `species_confidence`, not YOLO's label.

---

## Usage

**Run on images you already have:**

```bash
python3 scripts/main.py                       # everything in test_images/
python3 scripts/main.py path/to/image.jpg     # one image
python3 scripts/main.py some_folder/          # a folder
```

**Pull images from a Sage node, then process them:**

```bash
python3 scripts/pull_images.py --start -6h    # download into query_images/
python3 scripts/main.py query_images
```

`pull_images.py` defaults to node `H00F`. Use `--vsn H03B` for your own node.

### Useful flags

| Flag | Effect |
|---|---|
| `--no-hazard` | Skip Gemma. Much faster; no alerts. |
| `--baseline` | Only the first 5 images, into `output/baseline`. |
| `--output replace\|new\|<name>` | Where results go (skips the prompt). |
| `--min-species-confidence 0.5` | Raise/lower the identification floor. |
| `--confirmation-frames 3` | Frames an animal must appear in before it counts. |

---

## Output

Everything lands in `output/`:

- **`<name>_detected.jpg`** — the image with boxes and species labels.
  **Red** = dangerous, **green** = safe.
- **`detections.csv`** — one row per detection:

  `image, detected_as, yolo_confidence, species, common_name,
  species_confidence, hazard, danger_score, track_id, confirmed, x1, y1, x2, y2`

The CSV is rewritten each run. Load it with pandas for analysis:

```python
import pandas as pd
df = pd.read_csv("output/detections.csv")
df["common_name"].value_counts()
```

---

## Alerts

An alert fires when a track is **confirmed** *and* Gemma rates it
**dangerous** — once per animal, to every recipient in `ALERT_RECIPIENTS`,
plus a Slack post with the annotated image.

**Missing credentials don't stop detection.** If `twilio.env` is absent or
invalid, the pipeline logs `Twilio message failed` / `Slack message failed`
and keeps saving images and CSV rows.

---

## Deploying to a Sage node

Not in use yet — the nodes aren't deployed. When they are, `Dockerfile`,
`sage.yaml`, and `scripts/publisher.py` support running *on* a node, where the
plugin publishes detections to the Sage beehive (`--publish`) instead of
alerting directly. Nodes have restricted outbound networking, so an off-node
process would consume those measurements and send the alerts.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError` for a package that's in `requirements.txt` | Your venv is stale or inactive. Activate it, then re-run `pip install -r requirements.txt`. |
| `alert configuration unavailable` | `twilio.env` is missing or a field is malformed. Detection still runs. |
| Gemma download fails / 401 | Run `hf auth login` and accept the Gemma license. |
| `SAGE_USER and SAGE_USER_TOKEN are required` | Add them to `twilio.env`. Querying Sage is public; downloading images isn't. |
| Everything is slow | Models are on CPU. Check `torch.cuda.is_available()`. |
| SMS reaches one number but not another | Twilio trial — the other number isn't verified. |

---

## Project layout

```
danger-detection/
├── scripts/
│   ├── main.py          entry point
│   ├── pull_images.py   download images from Sage
│   ├── pipeline.py      ties the models together
│   ├── detector.py      YOLO
│   ├── species.py       BioCLIP
│   ├── hazard.py        Gemma
│   ├── tracker.py       same-animal tracking
│   ├── alerts.py        Twilio + Slack
│   └── publisher.py     Sage beehive publishing (node deployment)
├── test_images/         sample stills
├── test_sequences/      ordered frames for tracking
├── query_images/        pulled from Sage      (git-ignored)
├── output/              results               (git-ignored)
└── twilio.env           your credentials      (git-ignored)
```

---

The hazard rating describes a **species'** general capacity for harm — not the
behaviour of the individual animal in frame. BioCLIP can misidentify animals
and Gemma can be wrong, so treat alerts as triage, not ground truth.
