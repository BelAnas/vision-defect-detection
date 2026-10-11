"""Command line: inspect one image or a folder of images.

    python -m defects.cli data/samples
    python -m defects.cli part.jpg --json
    python -m defects.cli data/samples --save out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .service import Inspector, draw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="an image file or a folder of images")
    ap.add_argument("--json", action="store_true", help="print one JSON object per image")
    ap.add_argument("--save", type=Path, help="folder to write annotated images to")
    args = ap.parse_args(argv)

    if args.path.is_dir():
        paths = sorted(p for p in args.path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        paths = [args.path]
    if not paths or not all(p.exists() for p in paths):
        print(f"no image found at {args.path}", file=sys.stderr)
        return 2

    inspector = Inspector()
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
    unreadable = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path.name}: not a readable image", file=sys.stderr)
            unreadable += 1
            continue
        result = inspector.inspect(image)
        if args.json:
            print(json.dumps({"file": path.name, **result.to_dict()}))
        else:
            print(f"{path.name:28s} {result.verdict.upper():7s} {result.latency_ms:5.0f} ms  {result.reason}")
        if args.save:
            cv2.imwrite(str(args.save / f"{path.stem}.png"), draw(image, result))
    return 1 if unreadable else 0


if __name__ == "__main__":
    raise SystemExit(main())
