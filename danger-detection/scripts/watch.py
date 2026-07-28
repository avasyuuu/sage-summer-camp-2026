"""Continuously pull images from Sage, detect, alert, and clean up.

Runs as a long-lived process:

    pull new images into query_images/
        -> run the pipeline on each (safe / dangerous / nothing found)
        -> delete the processed image so the folder doesn't grow
        -> sleep N seconds
        -> repeat

Why a loop instead of cron: the models load **once** (~10 GB), and the tracker
keeps its state between cycles. A fresh process every few minutes would reload
everything and, worse, forget which animals it already alerted about - so an
animal that lingers in view would alert over and over.

Examples:
    python3 scripts/watch.py                     # poll H00F every 60s
    python3 scripts/watch.py --vsn H03B --interval 120
    python3 scripts/watch.py --no-pull           # process whatever is already
                                                 # in query_images/ (testing)
    python3 scripts/watch.py --once              # a single cycle, then exit
"""

import argparse
import csv
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline import CSV_FIELDS, IMAGE_EXTS, WildlifePipeline
from pull_images import (
    DEFAULT_OUT,
    download,
    load_sage_credentials,
    local_name,
    query_uploads,
)

_stop = False


def _handle_signal(signum, frame):
    """Finish the current image, then exit cleanly on Ctrl+C / systemd stop."""
    global _stop
    _stop = True
    print("\n[watch] stop requested; finishing the current cycle...", flush=True)


def log(message):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def fetch_new_images(vsn, start, limit, out_dir, auth):
    """Download any images we don't already have. Returns the new paths."""
    try:
        records = query_uploads(vsn, start)
    except Exception as exc:
        log(f"query failed: {type(exc).__name__}: {exc}")
        return []

    saved = []
    for record in records[-limit:]:
        url = record.get("value")
        if not url:
            continue
        dest = out_dir / local_name(record, url)
        if dest.exists():
            continue
        try:
            download(url, dest, auth)
            saved.append(dest)
        except Exception as exc:
            log(f"download failed ({url[-30:]}): {type(exc).__name__}: {exc}")
    return saved


def pending_images(out_dir):
    """Every image sitting in the folder, oldest first (chronological order)."""
    return sorted(
        p for p in out_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def append_rows(csv_path, rows):
    """Append to the CSV so results accumulate across cycles."""
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def process_one(pipeline, image_path):
    """Run one image. Returns (rows, summary string)."""
    detections = pipeline.process_image(image_path)
    if not detections:
        return [], "nothing detected"

    rows = pipeline._rows_for(image_path, detections)
    parts = []
    for det in detections:
        name = det.get("common_name") or det.get("species") or det["label"]
        if det.get("hazard"):
            parts.append(f"{name} [{det['hazard'].upper()} "
                         f"{det.get('danger_score', '?')}/10]")
        elif det.get("species"):
            parts.append(f"{name} [unassessed]")
        else:
            parts.append(f"{det['label']} [no species]")
    return rows, "; ".join(parts)


def run_cycle(pipeline, args, out_dir, auth):
    """One pull -> process -> delete pass."""
    if not args.no_pull:
        fetched = fetch_new_images(
            args.vsn, args.start, args.limit, out_dir, auth
        )
        if fetched:
            log(f"pulled {len(fetched)} new image(s)")

    images = pending_images(out_dir)
    if not images:
        return 0

    log(f"processing {len(images)} image(s)")
    processed = 0
    for image_path in images:
        if _stop:
            break
        try:
            rows, summary = process_one(pipeline, image_path)
            append_rows(pipeline.csv_path, rows)
            log(f"  {image_path.name}: {summary}")
            processed += 1
        except Exception as exc:
            log(f"  {image_path.name}: ERROR {type(exc).__name__}: {exc}")
        finally:
            # Delete either way: a file that always fails would otherwise be
            # retried forever and fill the disk.
            if not args.keep:
                try:
                    image_path.unlink()
                except OSError as exc:
                    log(f"  could not delete {image_path.name}: {exc}")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vsn", default="H00F", help="Sage node to pull from")
    parser.add_argument("--interval", type=int, default=60,
                        help="seconds to sleep between cycles (default: 60)")
    parser.add_argument("--start", default="-15m",
                        help="lookback window per pull (default: -15m)")
    parser.add_argument("--limit", type=int, default=25,
                        help="max images to pull per cycle (default: 25)")
    parser.add_argument("--no-pull", action="store_true",
                        help="skip downloading; process whatever is already in "
                             "query_images/ (useful for testing)")
    parser.add_argument("--keep", action="store_true",
                        help="do not delete processed images")
    parser.add_argument("--once", action="store_true",
                        help="run a single cycle and exit")
    parser.add_argument("--no-hazard", action="store_true",
                        help="skip Gemma (no hazard verdicts, no alerts)")
    parser.add_argument("--folder", default=str(DEFAULT_OUT),
                        help="image folder (default: query_images/)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    out_dir = Path(args.folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    auth = None
    if not args.no_pull:
        user, token = load_sage_credentials()
        if not (user and token):
            parser.exit(2, "SAGE_USER and SAGE_USER_TOKEN are required to pull "
                           "images. Use --no-pull to process a local folder.\n")
        auth = (user, token)

    # Alert config is optional: without it the pipeline logs the failure and
    # keeps detecting, which is what we want for an unattended service.
    alert_config = None
    try:
        from alerts import AlertConfigurationError, load_alert_config

        alert_config = load_alert_config(
            Path(__file__).resolve().parent.parent / "twilio.env"
        )
    except Exception as exc:
        log(f"alerts unavailable ({type(exc).__name__}); detection only")

    log("loading models (this takes a moment)...")
    pipeline = WildlifePipeline(
        use_hazard=not args.no_hazard,
        alert_config=alert_config,
    )
    log(f"ready. watching {'(local folder)' if args.no_pull else args.vsn} "
        f"every {args.interval}s; results -> {pipeline.csv_path}")

    total = 0
    while not _stop:
        try:
            total += run_cycle(pipeline, args, out_dir, auth)
        except Exception as exc:
            # Never let one bad cycle kill a long-running service.
            log(f"cycle failed: {type(exc).__name__}: {exc}")

        if args.once or _stop:
            break

        for _ in range(args.interval):      # responsive to Ctrl+C
            if _stop:
                break
            time.sleep(1)

    log(f"stopped after processing {total} image(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
