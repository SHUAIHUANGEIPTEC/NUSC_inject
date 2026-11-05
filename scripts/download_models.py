#!/usr/bin/env python
"""Download a few Objaverse .glb models into assets/models/.

Requires: pip install objaverse  (optional dependency)

Usage:
    python scripts/download_models.py --uids uid1 uid2 ...
    python scripts/download_models.py --uid-file path/to/uids.json [--n 3]

Picked UIDs should be license-checked first: Objaverse's per-object metadata
(metadata/{shard}.json.gz) carries a `license` field (by / cc0 / by-sa / by-nc /
by-nc-sa). Only `by`-class licenses are redistributed in assets/models/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", nargs="*", default=None)
    ap.add_argument("--uid-file", default=None,
                    help="JSON of Objaverse UIDs to sample from (omit with --uids)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "assets" / "models"))
    args = ap.parse_args()

    try:
        import objaverse
    except ImportError:
        sys.exit("objaverse not installed: pip install objaverse")

    uids = args.uids
    if not uids:
        if not args.uid_file:
            sys.exit("pass --uids, or --uid-file pointing at a JSON of Objaverse UIDs")
        uid_file = Path(args.uid_file)
        if not uid_file.exists():
            sys.exit(f"uid file not found: {uid_file}")
        data = json.loads(uid_file.read_text())
        flat = []
        for cls in data.values():
            if isinstance(cls, dict):
                for lst in cls.values():
                    if isinstance(lst, list):
                        flat.extend(lst)
                    elif isinstance(lst, str):
                        flat.append(lst)
            elif isinstance(cls, list):
                flat.extend(cls)
        uids = flat[: args.n]
        print(f"sampled {len(uids)} uids from {uid_file}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(uids)} objects ...")
    objs = objaverse.load_objects(uids)  # uid -> local glb path
    for uid, src in objs.items():
        dst = out / f"{uid}.glb"
        shutil.copy(src, dst)
        print("  ", dst)
    print("done.")


if __name__ == "__main__":
    main()
