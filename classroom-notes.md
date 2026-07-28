# Wildlife Detection Pipeline — Classroom Notes

*Last updated: 27 July 2026 · Work spanned 22–27 July 2026.*

Project: **`danger-detection`** (formerly `elk-grizzly-detection`) — a Sage camera
plugin that looks at wildlife
photos and answers three questions in order:

1. **Where is the animal?** → YOLO draws a bounding box.
2. **What species is it?** → BioCLIP identifies it.
3. **Is it dangerous?** → Gemma decides `safe` or `dangerous` and says why.

The output is an annotated image (box + label per animal) and one CSV of results.

---

## 1. The big picture: how the three models fit together

```
photo ──► YOLO ──► box + coarse label ──► BioCLIP ──► species + confidence ──► Gemma ──► safe/dangerous
          (find)     ("bear","dog"…)      (identify)   ("Ursus americanus")    (judge)   + one-line reason
```

Each model does one job, and each fixes the previous one's weakness:

- **YOLO** is great at *finding* animals but only knows 80 generic
  [COCO](https://cocodataset.org) classes — `bear`, `dog`, `horse`, `bird`… It
  has **no** class for elk, grizzly, raccoon, wolf, etc. So its label is often
  *wrong* (a raccoon comes back as `cat`, a wolf as `horse`). We only trust YOLO
  for the **box**, not the name.
- **BioCLIP** takes the cropped box and identifies the actual **species** from
  the biological "tree of life." This is what turns `horse` into
  *Canis lycaon* (wolf).
- **Gemma** (a small language model) reads the species + confidence and makes a
  **safety triage** call for a camera near people/trails.

> **Key lesson:** trust the *species confidence* from BioCLIP, not YOLO's label.
> YOLO mislabeling an animal is normal and expected — that's the whole reason
> BioCLIP is downstream.

### 1.1 The specific models we use (and sizes)

| Stage | Model in use | Notes |
|---|---|---|
| Detection | **YOLO11l** (large) | upgraded n → m → l for accuracy (23 Jul 2026) |
| Species ID | **BioCLIP** (tree of life) | identifies the species inside each box |
| Hazard | **Gemma 3 (`gemma-3-4b-it`)** | upgraded from 1b → 4b (~8.5 GB download) |

**YOLO11 comes in T-shirt sizes.** They're the *same* model scaled bigger — the
only trade-off is speed vs. accuracy:

| Model | Size | Parameters | Speed | Accuracy |
|---|---|---|---|---|
| YOLO11**n** | nano | ~2.6M | fastest | lowest |
| YOLO11**s** | small | ~9M | fast | better |
| YOLO11**m** | medium | ~20M | medium | good |
| YOLO11**l** | large | ~25M | slow | high ← **current** |
| YOLO11**x** | extra-large | ~57M | slowest | highest |

- **Bigger = more accurate but slower and heavier.** We moved up to `l` because
  trail-cam animals are often small or half-hidden, exactly where the small
  models miss things. On CPU it's slower; on the Sage Thor's GPU it's fine.
- Switching sizes is a one-line change: the model name in `detector.py` **and**
  `pipeline.py` (keep them matching!). New weights auto-download on first run
  (~50 MB for `l`, nothing like the 8.5 GB Gemma).
- The `.pt` weight files are git-ignored, so they don't get committed —
  teammates auto-download the same model on their first run.

---

## 2. Installation & setup

### 2.1 The most important lesson we learned

**Packages must be installed in the *same* Python you run the code with.**

A single computer often has several Pythons (Windows Store Python, Miniconda, a
venv, WSL's Python…). Typing `python` runs whichever one is first on your PATH.
If the packages were installed into Python A but your terminal runs Python B,
you get:

```
ModuleNotFoundError: No module named 'bioclip'
```

even though "nothing changed" in the code. We hit this exactly: packages were in
the Windows Store Python, but a Miniconda install had put conda's `base`
environment first on the PATH (via a `conda initialize` block in the PowerShell
profile), so `python` silently switched to a Python with nothing installed.

**Rules to avoid it:**

- Pick **one** interpreter (a venv or a conda env) and always use it.
- Install with `python -m pip install ...` (not bare `pip`) — the `-m` form
  guarantees the install lands in the interpreter you're running.
- Check which Python you're on with:
  ```bash
  python -c "import sys; print(sys.executable)"
  ```

### 2.2 Packages needed

All listed in `requirements.txt` (versions pinned so everyone installs the same):

| Package | Why |
|---|---|
| `pywaggle[vision]` | Sage plugin runtime; also brings OpenCV + NumPy |
| `ultralytics` | YOLO (the `yolo11n` model) |
| `pybioclip` | BioCLIP species identification |
| `transformers` | runs the Gemma language model (needs ≥ 4.50 for Gemma 3) |
| `accelerate` | helps `transformers` load/run the model |

Install:

```bash
python -m pip install -r requirements.txt
```

> **Torch is intentionally not pinned.** Each platform (and the Sage GPU base
> image) needs its own build of PyTorch, so we let `pip` resolve it.

### 2.3 Gemma model access (one-time)

Gemma is a *gated* model. Before the first run: make a Hugging Face account,
accept Google's Gemma license on the [model page](https://huggingface.co/google/gemma-3-1b-it),
then authenticate in your terminal:

```bash
hf auth login
```

### 2.4 First-run downloads

The model **weights** are not stored in the repo (they're large and, for Gemma,
license-gated). On the first run each downloads automatically into a local cache
(several GB total):

- `yolo11n.pt` (YOLO) — auto-downloads by name (~6 MB)
- BioCLIP — from Hugging Face
- Gemma — from Hugging Face (needs the license accepted above)

---

## 3. The scripts and what each does

All code lives in `scripts/`, one job per file:

| File | Class | What it does |
|---|---|---|
| `main.py` | — | Command-line entry point. Parses arguments, builds the pipeline, runs it. |
| `pull_images.py` | — | Downloads camera images from the Sage beehive into `query_images/`. |
| `pipeline.py` | `WildlifePipeline` | The conductor. Loads the models once, runs each image through every stage, draws boxes, writes the CSV. |
| `detector.py` | `AnimalDetector` | Wraps **YOLO**. `detect(image)` → boxes with a coarse label + confidence. |
| `species.py` | `SpeciesClassifier` | Wraps **BioCLIP**. Identifies the species in one crop. Auto-selects GPU when available. |
| `hazard.py` | `HazardClassifier` | Wraps **Gemma**. `assess(...)` → `safe`/`dangerous` + a 1–10 danger score. |
| `tracker.py` | `AnimalTrackRegistry` | Follows the same animal across frames so it alerts only once. |
| `alerts.py` | `AlertManager` + senders | Twilio SMS and Slack delivery. |
| `publisher.py` | `BeehivePublisher` | Publishes detections to Sage (only used when running *on* a node). |

### How a single image flows through `WildlifePipeline`

1. Read the image.
2. `AnimalDetector.detect()` → boxes.
3. Keep only boxes whose coarse label is an animal; hand them to the tracker,
   which assigns/updates a **track id** per animal.
4. For each track needing classification, `SpeciesClassifier` identifies the
   species from the best crop.
5. **Ignore any identification below 0.3 confidence** — BioCLIP always returns
   a best guess, and below that it is guessing.
6. `HazardClassifier.assess()` → safe/dangerous + score.
7. Draw boxes + labels, save the annotated image, collect CSV rows.
8. Confirmed **dangerous** tracks that haven't alerted yet → SMS + Slack.

After all images: write the single CSV.

---

## 4. How to run

Always run from the `danger-detection/` folder.

```bash
# process every image in test_images/ (the default)
python3 scripts/main.py

# process only specific images, or a folder
python3 scripts/main.py test_images/d70380s20i3.jpg
python3 scripts/main.py query_images

# skip the slow Gemma step while testing (no hazard, no alerts)
python3 scripts/main.py --no-hazard

# just the first 5 images, into output/baseline
python3 scripts/main.py --baseline
```

**The real workflow** — pull images from Sage, then process them:

```bash
python3 scripts/pull_images.py --start -6h    # into query_images/
python3 scripts/main.py query_images
```

`pull_images.py` defaults to node `H00F`; use `--vsn H03B` for our own node
once it is deployed.

---

## 5. Output

Everything lands in `output/` (git-ignored). Each run **overwrites**, so the
output always reflects the latest run.

### 5.1 Annotated images

One per input, named `<original>_detected.jpg`:

- A box around each detected animal.
- A label: the **common name** if BioCLIP has one, otherwise the **scientific
  name**, otherwise (only if BioCLIP found nothing) YOLO's coarse label.
- The box is **red** when Gemma flags it `dangerous`, **green** when `safe`.

### 5.2 The CSV — `output/detections.csv`

One row per detection (images with no animal get one blank-ish row so every
image is accounted for). Rewritten fresh each run — no duplicate rows.

| Column | Meaning |
|---|---|
| `image` | source filename |
| `detected_as` | YOLO's coarse COCO label (often "wrong" — that's expected) |
| `yolo_confidence` | how sure YOLO was there's *an animal* there |
| `species` | BioCLIP scientific name (e.g. `Ursus americanus`) |
| `common_name` | BioCLIP common name (may be blank for some taxa) |
| `species_confidence` | **BioCLIP's accuracy — the number to trust** |
| `hazard` | Gemma's verdict: `safe` or `dangerous` |
| `danger_score` | Gemma's 1–10 severity rating |
| `track_id` | which tracked animal this is (alerts fire once per track) |
| `confirmed` | whether the track met the confirmation threshold |
| `x1,y1,x2,y2` | box corner pixel coordinates |

Load it for analysis with pandas:

```python
import pandas as pd
df = pd.read_csv("output/detections.csv")
df["common_name"].value_counts()          # species tally
df[df["species_confidence"] >= 0.8]        # only trustworthy IDs
```

---

## 6. What the results told us (conclusions)

Running the pipeline over the 36 test images:

- **~21 of 36 images had a detectable animal**; the rest had none (empty frames
  are normal for trail cameras).
- **`species_confidence` cleanly separates good IDs from guesses.**

**High-confidence, correct identifications** (`species_confidence ≥ 0.9`):

| Species | Confidence | YOLO called it |
|---|---|---|
| Common raccoon (*Procyon lotor*) | 0.99 | `cat` |
| White-tailed deer (*Odocoileus virginianus*) | 0.98 | `sheep` |
| White-nosed coati (*Nasua narica*) | 0.97 | `bear` |
| American black bear (*Ursus americanus*) | 0.94–0.97 | `bear` |
| Sun bear (*Helarctos malayanus*) | 0.95 | `bear` |
| Yellow-throated marten (*Martes flavigula*) | 0.92 | `bird` |

**Low-confidence "noise"** (`species_confidence < 0.35`): the model returns an
obscure real species like "Greater mouse-deer 0.16" — this means *"I don't
know."* A confidence cutoff around **0.5** roughly separates real IDs from
noise.

**Showcase result — the wolf** (`d70380s20i3.jpg`): YOLO labeled it `horse`
(no wolf class in COCO), BioCLIP correctly identified *Canis lycaon* at 0.76,
and Gemma flagged it **DANGEROUS** — drawn with a red box. This one image shows
why all three stages are needed: YOLO alone would have called a wolf a horse.

> **Takeaway for analysis:** filter by `species_confidence`, not YOLO's
> `detected_as`. The mismatch between the two columns (raccoon↔cat, deer↔sheep,
> wolf↔horse) is the system working as designed, not a bug.

---

## 7. Problems we hit and how we fixed them (debugging log)

These are the real bugs we worked through — useful reference for next time.

### 7.1 `TritonMissing` when Gemma runs (Windows)
Gemma's `generate()` tries to `torch.compile` the model, whose backend needs
**Triton**, which has no Windows support. **Fix:** force plain (eager) execution
by disabling the compiler — set `TORCHDYNAMO_DISABLE=1` and
`torch._dynamo.config.disable = True` in `hazard.py`. Runs fine on CPU without it.

### 7.2 `top_p`/`top_k` "not valid" warning
Harmless. Those are *random-sampling* settings, but we decode **greedily**
(`do_sample=False`) for deterministic answers, so they're ignored. We cleared
them from the model's config to silence the warning. We do **not** need them —
a yes/no safety call should be deterministic, not random.

### 7.3 Images labeled "horse"/"dog" instead of the species
The drawing code only fell back to YOLO's label when BioCLIP's **common name**
was blank — but some real species (like *Canis lycaon*) have no common name.
**Fix:** label priority is now common name → **scientific name** → YOLO label,
so a blank common name shows *Canis lycaon* instead of `horse`.

### 7.4 "It worked before, now it doesn't" (`ModuleNotFoundError: bioclip`)
Not a code change — the terminal was running a different Python (Miniconda)
than the one with the packages. See §2.1. **Fix:** install into the interpreter
you actually run.

### 7.5 Does the Dockerfile handle installs?
Yes — but **only inside the container**, and only when you run `docker build` /
`docker run`. Typing `python scripts/main.py` runs your *local* Python and never
touches Docker. Docker is for deployment/consistency, not something the code
"points to" automatically.

### 7.6 Duplicate rows in the CSV
The old logger *appended* every run, so re-running stacked duplicate rows. **Fix:**
the CSV is now written **fresh** each run, so it's always exactly that run's data.

### 7.7 "Python not recognized" / phantom `.venv` (`ENOENT`)
VS Code's Python extension kept auto-creating a `.venv`, partially installing it,
then it got deleted — so the Run button pointed at a `.venv\Scripts\python.exe`
that no longer existed. **Fix:** rebuild the `.venv` properly (`python -m venv
.venv` + full `pip install -r requirements.txt`), then pin VS Code's interpreter
to it (§8.5). Underlying lesson: commit to **one** environment instead of letting
tools spawn new ones. See §8.

### 7.9 A folder rename silently disabled `.gitignore` (near-miss)
The root `.gitignore` used **anchored** paths:
`/elk-grizzly-detection/twilio.env`, `/elk-grizzly-detection/*.pt`. Renaming the
folder to `danger-detection/` made every one of those rules stop matching, so
the next commit swept in real Twilio credentials, a 50 MB model file, and 38
output images. Caught before pushing. **Fix:** use **unanchored** patterns
(`twilio.env`, `*.pt`, `output/`) that match in any folder and survive renames.
**Lesson:** after renaming a directory, run
`git check-ignore -v <path/to/secret>` to confirm the rules still bite.

### 7.10 `ModuleNotFoundError` for a package that IS in requirements
Slack alerts failed with `No module named 'slack_sdk'` even though
`slack-sdk==3.43.0` was listed. The venv had simply been created *before* that
line was added. **Lesson:** when an import fails for something requirements
clearly lists, the environment has drifted — re-run
`pip install -r requirements.txt` rather than debugging the code.

### 7.8 "It worked before through Docker" (it didn't)
A common misconception: believing `python main.py` ran through Docker. It never
did — that command runs local Python only. Docker requires explicit `docker
build`/`docker run`. See §8.2.

---

## 8. Environments — the mental model (READ THIS)

Most of our headaches came from environment confusion, not code. This section
is the antidote.

### 8.1 The one idea that explains everything

**Code only runs if the *specific Python you launch it with* has the packages
installed.** A computer can have several Pythons, and typing `python` picks
whichever is first on your PATH. If the packages live in Python A but your
terminal launches Python B, you get `ModuleNotFoundError` — even though "nothing
changed." That is the root cause of almost every problem below.

Check which Python you're on, anytime:

```bash
python -c "import sys; print(sys.executable)"
```

### 8.2 Local vs Docker are two SEPARATE paths

This is the big one. `python main.py` on your laptop **never uses Docker.**

| Path | What runs it | Needs a local venv/conda? |
|---|---|---|
| **Local** — `python main.py` | your laptop's Python | **Yes** — install `requirements.txt` |
| **Docker** — `docker build` + `docker run` | the container's own Python | No — the image is self-contained |

- Typing `python main.py` runs your **local** Python directly. Nothing reads the
  Dockerfile. To go through Docker you must explicitly run `docker build` then
  `docker run`.
- "When using Docker you don't need a Python environment" is TRUE — but only when
  you actually run *inside the container*. It does not apply to running
  `python main.py` locally.
- Proof it was never Docker: the `ModuleNotFoundError: bioclip` we hit. A
  container always has bioclip, so that error is impossible in Docker. It only
  happens when your *local* Python is missing the package.

So there are really three worlds, each with its own separate install:

| World | When it's used | Notes |
|---|---|---|
| **Local Python** (a venv) | `python main.py` on your laptop | Fastest for day-to-day development |
| **WSL (Linux on Windows)** | running in an Ubuntu terminal | Closer to the Sage node; Triton works here |
| **Docker container** | `docker build`/`docker run`; the Sage node | The deployment format; bundles everything |

### 8.3 venv vs conda — which to use locally

Use a **`.venv`** (a virtual environment) in the project folder. The rule of
thumb: **one isolated `.venv` per project.**

- A `.venv` holds only this project's packages, so projects can't break each
  other.
- conda `base` is a global, shared environment; installing project packages into
  it mixes everything together and can break other tools.
- VS Code auto-detects a `.venv` at the project root and activates it for you.
- It's the portable, standard recipe teammates can reproduce (see §2).

### 8.4 Activating `.venv` and getting out of conda

New PowerShell terminals may auto-start in conda's `(base)`. To switch to the
project venv:

```bash
conda deactivate                                   # leave conda (base)
& c:\...\sage-summer-camp-2026\.venv\Scripts\Activate.ps1   # enter .venv
```

Your prompt should then read just `(.venv)`. Verify with the `sys.executable`
check in §8.1 — it should show the `.venv\Scripts\python.exe` path.

- `deactivate` (no arguments) leaves the venv; `conda deactivate` leaves conda —
  different commands for different systems.
- To stop conda hijacking every new terminal: `conda config --set
  auto_activate_base false`. Then terminals open with no environment and you just
  activate `.venv`.

### 8.5 VS Code: pin the interpreter

If the Run button errors with a `...\.venv\Scripts\python.exe ... not
recognized` / `ENOENT`, VS Code is pointing at a deleted/rebuilt venv. Fix it
once: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the project's
`.venv`. VS Code stores the choice per-project; it is not part of git.

### 8.6 The Dockerfile base image (`waggle/sage-thor-base`)

`FROM waggle/sage-thor-base:0.1.0` is the base for the **Sage Thor** node
(NVIDIA Jetson Thor — ARM64 + CUDA). This is **correct** for deployment and
should be left as-is.

- It's only awkward if you try to `docker build`/`docker run` it **locally** on a
  Windows/x86 laptop: it's a large ARM/CUDA image and would need emulation or run
  CPU-only.
- You don't need to build it locally. Develop with the `.venv`; the Dockerfile
  runs on the actual Sage Thor. The two live on different machines and don't
  conflict.

### 8.7 Reproducibility for the team

For a project multiple people run, **Docker (or a documented venv + pinned
`requirements.txt`) is what makes it portable** — "does your Python have the
packages?" stops being a question because the answer is either baked into the
image or spelled out in the setup steps. "It worked on my machine" almost always
means the packages were installed into one Python by hand and never written down.

---

## 9. Known limitations & next steps

- **Our nodes have no camera yet.** H03B and H02C are online and publish system
  metrics (CPU, thermal, network) but **zero images**. Until a camera plugin is
  scheduled on them there is nothing of ours to detect on — we pull from H00F
  instead. *Open question for the instructor, along with the capture interval.*
- **Duplicate alerts.** YOLO sometimes draws two boxes on one animal, creating
  two tracks and therefore two alerts for the same wolf. Needs box merging or
  stricter matching.
- **Constrained species list:** BioCLIP searches the *entire* tree of life.
  Giving it a fixed list of target species would cut noise further than the
  confidence floor alone. Supported via `species_labels=[...]`.
- **Capture time:** the pipeline records processing time, not photo time. For
  time-of-day analysis, read it from the image metadata.
- **SAM 3** was explored early but its weights are license-gated and access
  never came through, so it was removed.

---

## 9.5 Where things actually run (current architecture)

This changed a lot, so it is worth stating plainly.

```
Sage node (camera)  ──uploads images──►  Sage beehive
                                              │
                            pull_images.py ───┘
                                              ▼
                                    query_images/  (on the DGX Spark)
                                              │
                              YOLO → BioCLIP → Gemma   (all on the Spark)
                                              ▼
                                    Twilio SMS + Slack
```

**Everything runs on one machine we control** (a DGX Spark). Sage is only the
**image source**. Reasons:

1. **Nodes have restricted outbound networking** — a node cannot reach Twilio
   or Slack directly, so alerting from the node was never going to work.
2. **Gemma is big.** The BF16 4B checkpoint is ~8 GB; that does not fit an 8 GB
   Xavier NX at all, and even on bigger hardware it's heavy for an edge box.
3. **Credentials stay off the nodes entirely** — no secrets in the repo, in the
   container, or on Sage. This removed a whole class of deployment problems.

**Consequence:** the Sage app / `Dockerfile` / `publisher.py` path is built but
**not in use**. It exists for later, when nodes are deployed and we might want
detection running on-node (publishing to the beehive instead of alerting).

### The DGX Spark setup (what mattered)

- GPU is an **NVIDIA GB10** (Grace Blackwell, `sm_121`), CUDA 13.0.
- `pip install -r requirements.txt` pulled the right build automatically
  (`torch 2.13.0+cu130`) — no manual index URL needed.
- **On Blackwell, always test a real matmul**, not just
  `torch.cuda.is_available()`. `is_available()` can report `True` while kernels
  fail with *"no kernel image is available for execution on the device"* if the
  torch build predates the GPU's compute capability.
- **BioCLIP needed an explicit device.** pybioclip defaults to `device='cpu'`,
  so it silently stayed on the CPU while YOLO and Gemma used CUDA.
  `SpeciesClassifier` now auto-selects `cuda` and falls back to `cpu`, so the
  same code is fast on the Spark and still works on a Mac.

---

## 10. SMS alerts & object tracking

Two features layered on top of the detection pipeline (late July 2026).

### SMS alerts (Twilio)
- When a **dangerous** animal is confirmed, the pipeline texts an alert via Twilio.
- Credentials live in `twilio.env` (git-ignored): `TWILIO_ACCOUNT_SID`,
  `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `ALERT_RECIPIENTS` (E.164,
  comma-separated). The app loads it automatically via `python-dotenv`.
- **Missing credentials no longer stop the run.** If `twilio.env` is absent or
  invalid, the pipeline logs `Twilio message failed` / `Slack message failed`
  and keeps saving images and CSV rows. (It used to hard-exit at startup; that
  made the code undeployable anywhere without credentials.)
- Slack posts the annotated image alongside the text, via `slack_sdk`.
- **Trial-account limits:** it can only text **verified** numbers and can't use a
  custom message body (sends a fixed template). A paid account lifts both.

### Object tracking
- The pipeline tracks **individual animals across frames**, not just species, so
  each new dangerous animal alerts **once** — even if it lingers in view for hours.
- Confirmation now defaults to **1 frame** (it was 3 early on), so a single
  still *does* classify and alert. Raise it with `--confirmation-frames 3` if
  one-frame false alarms become a problem. A track is dropped after **30 missed
  frames**; a returning animal counts as new.
- The matcher (`tracker.py`, `AnimalTrackRegistry`) is tuned for sparse
  fixed-camera stills — it survives YOLO label flips (e.g. bear→cow→bear) and
  needs the `lap` library.

### Deferred for later
A dual-camera setup (paired visible + thermal, sometimes a side-by-side composite
image) with cross-camera sensor fusion was designed but **not built yet** — v1
assumes one fixed visible-light feed.

---

## 11. Timeline

**22 July 2026**
- Built the YOLO detection backbone; switched to the YOLO11 model.
- Fixed an empty `detector.py` in the repo; added the `AnimalDetector` class.
- Added BioCLIP species identification on the cropped detections.
- Added a SAM 3 pipeline (parked — weights are license-gated, access pending).
- Added CSV logging of detections.
- Reorganized the project into `scripts/` and `output/` folders.

**23 July 2026**
- Reviewed an external `detectors.py` reference (YOLO + BioCLIP backends).
- Wired in Gemma 3 hazard assessment (`safe`/`dangerous`).
- Fixed the `TritonMissing` crash, the `top_p`/`top_k` warning, and the
  "horse"/"dog" mislabeling (see §7).
- Consolidated 8 files into 5 clean classes with a single results CSV.
- Pinned `requirements.txt` and wrote the README for reproducibility.
- Worked through the local-environment setup (venv vs. conda vs. Docker) and
  settled on a project `.venv` (see §8).
- Added a startup prompt for where results go: replace output, new output
  folder, or a baseline run (first 5 images into `output/baseline`).
- Upgraded the Gemma model from `gemma-3-1b-it` to `gemma-3-4b-it`.
- Upgraded YOLO from nano → medium → **large** (`yolo11l`) for better accuracy;
  kept `detector.py` and `pipeline.py` in sync (see §1.1).
- Fixed a merge break: `main.py` still passed a removed `context` argument to the
  species-only hazard classifier (`TypeError`).

**26–27 July 2026**
- Added Twilio SMS alerts for confirmed dangerous animals (mandatory `twilio.env`).
- Added single-camera object tracking (3-frame confirmation, 30-frame retirement)
  so each animal alerts once; dropped an earlier per-species cooldown idea.
- Reverted Gemma from the Q4_0 GGUF / `llama-cpp-python` runtime back to BF16
  `gemma-3-4b-it` via Transformers (the Thor has enough memory for it).
- Deferred dual-camera / thermal fusion to a later phase (see §10).

**27 July 2026**
- Renamed the project `elk-grizzly-detection` → `danger-detection`. The old
  `.gitignore` rules were anchored to the old folder name, so the rename
  silently disabled them and swept `twilio.env` into a commit — caught before
  pushing (see §7.9).
- Added `pull_images.py`: download node camera images from the beehive.
- **Architecture pivot** — everything now runs on the DGX Spark, with Sage as
  the image source only (see §9.5).
- Added a no-credentials failsafe so detection runs without Twilio/Slack.
- Fixed BioCLIP silently running on CPU on a GPU machine.
- Added a **0.3 confidence floor** on species identification, so garbage guesses
  (0.02, 0.09) no longer reach Gemma or trigger alerts.
- Removed the separate off-node watcher; its query and alert code already lived
  in `pull_images.py` and `alerts.py`.
- Set up the Spark end to end: GPU verified, full pipeline, real SMS + Slack.
