# Wildlife Detection Pipeline — Classroom Notes

*Last updated: 23 July 2026 · Work spanned 22–23 July 2026.*

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

### 7.7 "Python not recognized" / phantom `.venv` (`ENOENT`)
VS Code's Python extension kept auto-creating a `.venv`, partially installing it,
then it got deleted — so the Run button pointed at a `.venv\Scripts\python.exe`
that no longer existed. **Fix:** rebuild the `.venv` properly (`python -m venv
.venv` + full `pip install -r requirements.txt`), then pin VS Code's interpreter
to it (§8.5). Underlying lesson: commit to **one** environment instead of letting
tools spawn new ones. See §8.

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

---

## 10. Timeline

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
