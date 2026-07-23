# Wildlife Detection Pipeline — Classroom Notes

Project: **`elk-grizzly-detection`** — a Sage camera plugin that looks at wildlife
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

All code lives in `scripts/`. It was consolidated from 8 messy files into 5
clean classes, each with one job:

| File | Class | What it does |
|---|---|---|
| `main.py` | — | Command-line entry point. Parses arguments, builds the pipeline, runs it. |
| `pipeline.py` | `WildlifePipeline` | The conductor. Loads the 3 models once, runs each image through all stages, draws the boxes, and writes the CSV. |
| `detector.py` | `AnimalDetector` | Wraps **YOLO**. `detect(image)` → list of boxes with a coarse label + confidence. |
| `species.py` | `SpeciesClassifier` | Wraps **BioCLIP**. `classify(image, box)` → species name, common name, and confidence for one crop. |
| `hazard.py` | `HazardClassifier` | Wraps **Gemma**. `assess(...)` → `safe`/`dangerous` + a one-line reason. |

### How a single image flows through `WildlifePipeline`

1. Read the image.
2. `AnimalDetector.detect()` → boxes.
3. For each box whose label is an animal, `SpeciesClassifier.classify()` →
   species.
4. If a species was found, `HazardClassifier.assess()` → safe/dangerous.
5. Draw the boxes + labels and save the annotated image.
6. Collect one CSV row per detection.

After all images: write the single CSV.

---

## 4. How to run

Always run from the `elk-grizzly-detection/` folder.

```bash
# process every image in test_images/ (the default)
python scripts/main.py

# process only specific images
python scripts/main.py test_images/d37363s15i5.jpg test_images/d70380s20i3.jpg

# tell Gemma the real setting (improves the safety judgment)
python scripts/main.py --context "Camera beside a campground in northern Wisconsin"

# skip the slow Gemma step while testing (hazard columns left blank)
python scripts/main.py --no-hazard
```

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
| `reason` | Gemma's one-line justification |
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

---

## 8. Environments — the mental model

Three separate "worlds," each needing its own package install:

| World | When it's used | Notes |
|---|---|---|
| **Local Python** (venv / conda) | `python scripts/main.py` on your machine | Fastest for development. Must install `requirements.txt`. |
| **WSL (Linux on Windows)** | running in an Ubuntu terminal | Closer to the Sage node; Triton works here. Separate install. |
| **Docker container** | `docker build` + `docker run`; the Sage node | The Dockerfile installs everything inside the image. This is the deployment format. |

For a project multiple people run, **Docker (or a documented venv setup) is what
makes it portable** — "does your Python have the packages?" stops being a
question because the answer is baked into the image.

---

## 9. Known limitations & next steps

- **SAM 3** (text-prompt detection, an alternative to YOLO) was explored but its
  weights are license-gated on Hugging Face and access was still pending, so it's
  not in the current pipeline.
- **Constrained species list:** BioCLIP currently searches the *entire* tree of
  life. Giving it a fixed list of target species (elk, grizzly, etc.) would kill
  most of the low-confidence noise. Supported via `species_labels=[...]`.
- **Gemma reasoning** is occasionally muddled (e.g. calling a 0.89-confidence ID
  "low confidence"). Tunable via the prompt in `hazard.py`.
- **Capture time:** the pipeline records processing time, not photo time. For
  time-of-day analysis, read it from the image later.
