# SMS Alert System Design Notes

Last updated: July 26, 2026 (America/Chicago)

## Purpose

Add SMS warnings to the elk/grizzly detection pipeline when it identifies a
dangerous animal. These notes preserve the design conversation and current
implementation state so work can continue across sessions.

## Current pipeline

The existing image pipeline is:

1. YOLO locates animals and produces bounding boxes.
2. BioCLIP identifies the species.
3. Gemma assigns a binary `safe` or `dangerous` label and a danger score from
   1 through 10.
4. Annotated images and `output/detections.csv` are written.

The current hazard classification describes a species' general capacity to
harm humans. It does not establish that an individual animal is behaving
aggressively or presents an immediate emergency. Alert wording should therefore
say that a dangerous species was detected.

## Twilio connectivity work

Twilio is installed through `requirements.txt` as `twilio==9.10.9`.

Configuration is read from environment variables:

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_NUMBER`
- `ALERT_RECIPIENTS` (comma-separated, E.164 format)
- `ALERTS_ENABLED=true`

`.env` and `twilio.env` are ignored by Git. Credentials and phone numbers must
not be committed or copied into this document.

The standalone command `python scripts/alerts.py --send-test` was retained for
now. Twilio authentication, sender, recipient, and delivery were successfully
tested with a real SMS.

The Twilio account is currently a trial account. Trial SMS requests cannot use
a custom body, so the code sends the predefined `sms_internal_alerts` template.
The intended custom wildlife alert is printed locally. A paid account will be
needed to send the custom wildlife message body.

## Alert integration already implemented

`AlertManager` in `scripts/alerts.py` is connected to `scripts/pipeline.py`
after all detections for one image have been classified. This placement avoids
sending one SMS per bounding box when several dangerous detections occur in one
image.

`scripts/main.py` currently supports:

- No alert flag: alerts are off.
- `--alert-dry-run`: print the intended alert without contacting Twilio.
- `--alerts`: send a real Twilio trial-template SMS.

Twilio failures are reported without stopping annotated-image or CSV output.
The dry-run and integrated real-SMS paths were successfully tested using:

`infrared_images/imagesampler-mobotix-2689__1777723214884804333-sample.jpg`

## Cooldown implementation and rejected design

A configurable, in-memory, per-species cooldown was implemented using
`--alert-cooldown-minutes` with a default of 10 minutes. It passes local tests,
but this is no longer the desired product behavior and should be removed or
replaced when object tracking is implemented.

Reason for rejection: a dangerous animal could remain in view for many hours.
A time-based species cooldown would either send repetitive alerts for the same
stationary animal or suppress a genuinely new animal of the same species.

Do not proceed with the previously proposed persistent species-cooldown database.

## Agreed object-tracking behavior

The desired design tracks individual detected animals, not merely species:

1. Every new dangerous animal object should generate an SMS alert.
2. Once that tracked object generates an alert, mark that object as alerted.
3. Do not send another alert for that same active tracked object, regardless of
   how long it remains visible (including an animal sleeping in view for many
   hours).
4. Multiple animals may be active simultaneously and must have distinct track
   identities.
5. A newly appearing dangerous animal should generate a new alert even when a
   different animal was detected recently. The proposed global ten-minute rate
   limit has been removed.
6. If an animal is unmatched for 30 consecutive processed frames, consider it
   gone and remove its track from memory.
7. If an animal returns after its track was removed, treat it as a new object
   and allow a new alert.
8. Absence will be measured in frames, not seconds. Every processed camera
   image must advance tracker state, including images with no detections, only
   safe animals, or failed classification.
9. A new object must be matched in at least three consecutive processed frames
   before it is confirmed and eligible for classification/alerting. A miss
   before confirmation resets its consecutive-seen count. This confirmation
   rule reduces one-frame false alerts.

The camera image interval is not currently known. The user expects to provide
it on July 27, 2026. Thirty missed frames remains the chosen rule, but the image
interval is needed to understand how much wall-clock time it represents.

## Camera design

The intended system uses two fixed cameras. Neither camera moves. The current
understanding is that both cameras observe the same scene: one is a conventional
visible-light camera and the other is thermal/infrared for stronger nighttime
detection. This should be confirmed against the actual installation.

An example dual-camera screenshot was reviewed on July 26, 2026. It is a single
wide, side-by-side composite: false-color thermal imagery occupies the left
portion and visible-light imagery occupies the right portion. This suggests
periodic input may arrive as one composite image rather than two separate image
files. The ingestion layer should detect or be configured for this layout and
split each composite into its thermal and visible views before detection.

The reviewed example is approximately 1556 by 506 pixels with a vertical join
near the center, but production dimensions and the exact split coordinate must
be confirmed from original camera output rather than hard-coded from a
screenshot.

The two example views cover much of the same physical scene but are not
pixel-aligned. Foreground equipment and a person appear at substantially
different horizontal positions and scales, indicating viewpoint/parallax and
possibly different fields of view. Consequently, a thermal box and a visible
box cannot be merged merely because they share the same pixel coordinates.
The fixed installation is still suitable for a one-time geometric calibration,
but nearby foreground objects may require a richer mapping than one global
homography.

The user would like an animal visible to both cameras to be represented as one
object so it does not generate duplicate alerts. Because these appear to be two
modalities of the same scene, the preferred design is visible/thermal sensor
fusion rather than two unrelated trackers. Ordinary independent camera-local
tracking IDs would still be insufficient by themselves.

If frames are captured at approximately the same time, detections can be mapped
into one shared scene using a one-time geometric calibration between the fixed
cameras. Overlapping projected boxes or nearby projected centers can then be
treated as observations of the same global object. A single fused global track
would preserve one `alerted` flag even if the object is observed by both sensors.

At night the thermal camera may be the only sensor with a reliable detection.
The system must allow a global track to continue from either modality alone;
it cannot require both cameras to see the animal on every frame. Species
classification from thermal imagery may also be less reliable than from visible
imagery, so the eventual design should prefer a good visible-light crop for
BioCLIP when available and explicitly test the thermal-only path.

General cross-camera approaches depend on the physical installation:

- Overlapping views with known camera geometry: match detections using time,
  calibrated scene position, and appearance.
- Adjacent/non-overlapping views: hand off identities using exit/entry regions,
  timestamps, expected travel time, species, and appearance embeddings.
- Unrelated views: robust identity matching is much harder; treating tracks as
  camera-local may be safer than incorrectly merging two different animals.

Fixed cameras make cross-camera association more feasible, but two animals of
the same species can still look similar, lighting can differ, and occlusion can
cause identity changes. A tracker ID is a temporary software identity, not a
guaranteed biological identity.

## Likely implementation direction

Use a tracker such as ByteTrack or BoT-SORT to associate YOLO boxes across
chronological frames. Each active track should retain at least:

- Camera-local track ID
- Global/cross-camera ID when association is sufficiently reliable
- Latest bounding box
- YOLO class
- BioCLIP species and confidence
- Hazard label and danger score
- Whether an alert was sent
- Consecutive missed-frame count
- First-seen and last-seen image/timestamp
- Camera ID

The precise cross-camera strategy must be selected after learning the camera
layout and image metadata.

## Open questions for the next session

1. How often does each camera save an image?
2. Do the two camera views overlap? If so, approximately how much?
3. Are the cameras synchronized, or how far apart can their capture times be?
4. Do image filenames or metadata contain reliable capture timestamps?
5. Can each image be reliably attributed to camera 1 or camera 2?
6. Are the cameras observing the same physical area from different angles, or
   separate/adjacent areas?
7. Is there existing calibration information or a site map showing camera
   positions and fields of view?
8. Should several new dangerous animals appearing in one frame produce one
   combined SMS or one SMS per animal? The current `AlertManager` sends at most
   one message per image, so this must be decided before tracking integration.
9. Are visible and thermal images captured simultaneously or at different
   intervals, and how accurately can they be paired by timestamp?
10. Are the two cameras mounted close together and aligned, or do they observe
    the same area from meaningfully different angles?
11. What are the resolution and field of view of each camera?
12. Does the thermal camera produce radiometric thermal data, grayscale images,
    or false-color images?
13. Does every production capture arrive as one side-by-side composite like the
    reviewed example, and is the split position constant?
14. Can several original paired captures (not screenshots) be obtained with an
    animal or person visible in both views for calibration and matching tests?

## Immediate next step

### Current implementation scope decision

On July 26, 2026, the initial tracker scope was deliberately narrowed. For now,
assume one consistent, fixed, non-thermal image feed that BioCLIP can classify.
Do not implement dual-camera splitting, calibration, thermal classification, or
cross-camera identity fusion in the first version. Preserve the dual-camera
research above for a later phase.

The immediate next implementation is to replace the species cooldown with
single-camera track-based alert state. Process the input images as chronological
frames, assign persistent object IDs, alert once for each new dangerous track,
confirm a new track after three consecutive observed frames, and remove a track
after 30 consecutive frames without a matching observation.

For the first version, retain the best animal crop (highest YOLO confidence)
from the three confirmation frames. Once the track reaches its third consecutive
frame, run BioCLIP and Gemma on that best crop. If the result is dangerous, send
one alert and mark the track alerted. An unconfirmed track never sends an alert.

### First implementation completed

The first single-camera tracking iteration was implemented on July 26, 2026:

- `scripts/detector.py` uses persistent Ultralytics ByteTrack IDs.
- `scripts/tracker.py` implements three-frame confirmation, best-crop retention,
  per-object classification/alert state, and removal after 30 missed frames.
- `scripts/pipeline.py` classifies confirmed objects and routes dangerous,
  unalerted tracks to Twilio.
- `scripts/alerts.py` no longer uses the rejected species cooldown; it marks a
  track alerted after a successful delivery (or a dry-run simulation) and
  permits retry after complete delivery failure.
- `scripts/main.py` exposes `--confirmation-frames` and
  `--max-missed-frames`; `--alert-cooldown-minutes` was removed.
- `tests/test_tracking_alerts.py` covers confirmation, streak reset, multiple
  objects, retirement, reappearance, safe tracks, one-time delivery, and retry.

ByteTrack requires the lightweight `lap==0.5.12` assignment library. It was
added to `requirements.txt`; it is a Python dependency, not a new ML model.
Nine automated state/alert tests pass. A real YOLO/ByteTrack smoke test retained
two stable animal IDs across three repeated frames.

### First real sequence results

A real fixed-camera sequence was tested on July 26, 2026. Frames 1 through 3
retained ByteTrack ID 1, reached the three-consecutive-frame confirmation rule,
and produced a real Twilio SMS when run with `--alerts`. YOLO used the coarse
COCO label `bear`; BioCLIP identified the spotted animal as `Panthera pardus`
(Leopard) with 0.9602 confidence; Gemma classified it dangerous at 8/10.

The isolated/random animal images later in the sequence did not reach three
consecutive tracked observations and did not alert, as intended.

ByteTrack lost ID 1 in frames 4 and 5 even though YOLO continued detecting the
same visible animal (`cow` at 0.4054, then `bear` at 0.792). The bounding box
moved and changed size enough that default video-oriented association did not
pass. The application registry retained old track 1 with increasing missed-frame
counts, but the untracked observations could not update it.

The next engineering priority is sparse-still association: preserve one object
identity across larger inter-frame movement and changes among YOLO's coarse
animal labels. Success criterion: all five known same-animal frames share one
application track, produce one confirmation and one alert, while isolated
animals remain unconfirmed.

### Sparse-frame association implemented

On July 27, 2026, ByteTrack identity assignment was replaced with an
application-level matcher designed for fixed-camera periodic stills. YOLO now
returns raw boxes. `AnimalTrackRegistry` assigns IDs using predicted center
motion, normalized center distance, box size, overlap, HSV crop appearance, and
one-to-one LAP assignment. Changes among YOLO's coarse animal classes do not
break identity.

A fixed-camera scene-consistency gate compares small normalized grayscale scene
signatures. It prevents unrelated backgrounds in the mixed test folder from
being chained into one track. This gate assumes the first-version scope of one
fixed camera; substantial viewpoint changes intentionally create new tracks.

Validation on the full test folder produced the intended result:

- All five consecutive leopard frames remained application track 1, including
  the `bear -> cow -> bear` YOLO label changes in frames 3 through 5.
- Track 1 confirmed exactly once on frame 3.
- Every unrelated/isolated animal image received a new candidate ID with a
  consecutive count of 1.
- No unrelated track became confirmed.
- Twelve automated tracking/alert tests pass, including sparse movement, label
  changes, two simultaneous moving animals, scene separation, retirement,
  reappearance, and delivery retry behavior.

## Initial tracking test data

The user has collected five images of a bear from the same fixed camera view,
with the bear in different positions. These can form the first positive tracking
sequence if they are ordered chronologically and the animal's displacement
between adjacent frames is not unrealistically large.

Do not interleave unrelated animal photographs from different cameras or
backgrounds into this sequence. The tracker treats every file as the next frame
from one fixed camera, so arbitrary scene changes do not represent the intended
deployment and can create misleading track resets or new IDs.

Use separate test sequences:

- Five ordered same-camera bear frames: expect confirmation on frame three and
  exactly one alert across all five frames.
- One or two bear frames followed by a same-camera empty frame: expect no alert
  because the three-frame confirmation threshold was not reached.
- Same-camera negative/empty frames: check that YOLO does not create false
  animal tracks.
- Separate safe-animal sequences: check that confirmed safe tracks do not alert.
- Thirty consecutive same-camera frames without the bear, followed by its
  return: test track retirement and new alert eligibility. Until 30 real empty
  frames are available, automated registry tests can simulate them without
  duplicating image files.

Unrelated random animal images remain useful for detector/species/hazard
regression testing, but they should be run separately from the fixed-camera
object-tracking sequence.

## Final alert startup decisions (July 27, 2026)

The deployment direction was simplified so alert delivery is always enabled.
The alert-off and dry-run modes were removed, including the `--alerts`,
`--alert-dry-run`, and `ALERTS_ENABLED` controls. A confirmed dangerous track
automatically attempts SMS delivery. Missing or invalid Twilio configuration
causes a clear startup failure rather than silently disabling alerts.

`twilio.env` is loaded automatically by the Python application; operators do
not run `source twilio.env`. `python-dotenv` resolves the file relative to the
project/application root rather than the current working directory. Startup
immediately validates `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`,
`TWILIO_FROM_NUMBER`, and `ALERT_RECIPIENTS` before loading ML models.

For local execution, placing the ignored `twilio.env` in the project root allows
`python3 scripts/main.py` to work without shell setup. `.dockerignore` prevents
the credentials from entering the image. For containerized deployment,
provision the protected file once on the node and mount it read-only at
`/app/twilio.env`. The existing container entry point starts `main.py`, which
loads and validates the file automatically on every container start or restart.

## Gemma runtime transition (July 27, 2026)

The Jetson Thor deployment has enough unified memory to run the standard Gemma
3 4B BF16 checkpoint without four-bit quantization. The hazard classifier was
therefore changed from the Q4_0 GGUF checkpoint and `llama-cpp-python` runtime
to `google/gemma-3-4b-it` through Hugging Face Transformers, Accelerate,
PyTorch, and CUDA. The model is configured with `GEMMA_MODEL_ID` or
`--gemma-model-id`; the old repository-plus-GGUF-filename settings were removed.

Gemma initialization failures now stop pipeline startup instead of silently
disabling hazard assessment. The model requires prior acceptance of Google's
Gemma license and Hugging Face authentication for its initial download. The
model cache should be mounted on persistent node storage before deployment so
container restarts do not repeat the approximately 8.6 GB download.
