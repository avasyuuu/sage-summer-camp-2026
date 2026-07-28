"""Download camera images from the Sage beehive into a local folder.

Sage nodes upload images to the beehive; this pulls them down so the detection
pipeline can run wherever you have the horsepower (e.g. a DGX), instead of on
the node itself.

    beehive (node uploads)  --> pull_images.py --> query_images/
                                                        |
                                            main.py query_images/  --> alerts

Querying the beehive is public, but DOWNLOADING an image needs Sage
credentials. Set them in the environment or in twilio.env:

    SAGE_USER=<your sage username>
    SAGE_USER_TOKEN=<your sage token>

Examples:
    python3 scripts/pull_images.py                    # last hour from H00F
    python3 scripts/pull_images.py --vsn H03B         # your node, once deployed
    python3 scripts/pull_images.py --start -6h --limit 50
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "query_images"
SAGE_QUERY_URL = "https://data.sagecontinuum.org/api/v1/query"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_sage_credentials(env_path=PROJECT_ROOT / "twilio.env"):
    """Read Sage credentials from the environment, falling back to twilio.env."""
    user = os.environ.get("SAGE_USER", "").strip()
    token = os.environ.get("SAGE_USER_TOKEN", "").strip()
    if user and token:
        return user, token

    env_path = Path(env_path)
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" not in line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if key.strip() == "SAGE_USER" and not user:
                user = value
            elif key.strip() == "SAGE_USER_TOKEN" and not token:
                token = value
    return user, token


def query_uploads(vsn, start, timeout=60):
    """Return upload records (newest last) for one node."""
    payload = {"start": start, "filter": {"vsn": vsn, "name": "upload"}}
    response = requests.post(SAGE_QUERY_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def local_name(record, url):
    """Build a chronologically sortable filename.

    The tracker treats files in sorted order as consecutive frames, so the
    timestamp must lead the name for the sequence to make sense.
    """
    stamp = str(record.get("timestamp", ""))
    # 2026-07-27T18:04:05.123456789Z -> 20260727T180405
    cleaned = (
        stamp.replace("-", "").replace(":", "").split(".")[0].replace("Z", "")
    ) or datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = Path(url).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"
    return f"{cleaned}_{Path(url).name[:12]}{suffix}"


def download(url, dest, auth, timeout=120):
    response = requests.get(url, auth=auth, timeout=timeout)
    if response.status_code == 401:
        raise PermissionError(
            "401 from the object store - check SAGE_USER / SAGE_USER_TOKEN"
        )
    response.raise_for_status()
    dest.write_bytes(response.content)
    return len(response.content)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vsn", default="H00F",
                        help="node to pull from (default: H00F, a node that "
                             "currently publishes images)")
    parser.add_argument("--start", default="-1h",
                        help="lookback window, e.g. -30m, -6h, -2d (default: -1h)")
    parser.add_argument("--limit", type=int, default=25,
                        help="max images to download (default: 25)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="destination folder (default: query_images/)")
    parser.add_argument("--clean", action="store_true",
                        help="empty the destination folder first")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        removed = 0
        for path in out_dir.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
        print(f"cleaned {removed} file(s) from {out_dir}")

    user, token = load_sage_credentials()
    if not (user and token):
        parser.exit(2, "SAGE_USER and SAGE_USER_TOKEN are required to download "
                       "images (querying is public, downloading is not).\n")
    auth = (user, token)

    try:
        records = query_uploads(args.vsn, args.start)
    except Exception as exc:
        parser.exit(1, f"query failed: {type(exc).__name__}: {exc}\n")

    if not records:
        print(f"no images found for {args.vsn} in window {args.start}")
        return

    # Newest first, then trim, so --limit keeps the most recent frames.
    records = records[-args.limit:]
    print(f"{args.vsn}: {len(records)} image(s) to fetch into {out_dir}")

    saved = skipped = failed = 0
    for record in records:
        url = record.get("value")
        if not url:
            continue
        dest = out_dir / local_name(record, url)
        if dest.exists():
            skipped += 1
            continue
        try:
            size = download(url, dest, auth)
            saved += 1
            print(f"  saved {dest.name} ({size} bytes)")
        except Exception as exc:
            failed += 1
            print(f"  FAILED {url[-40:]}: {type(exc).__name__}: {exc}")

    print(f"\ndone: {saved} saved, {skipped} already present, {failed} failed")
    if saved or skipped:
        print(f"run the pipeline with:\n  python3 scripts/main.py {out_dir}")


if __name__ == "__main__":
    main()
