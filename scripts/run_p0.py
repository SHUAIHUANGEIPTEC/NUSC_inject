#!/usr/bin/env python
"""P0 single-sample / single-object entry (uses the shared pipeline).

Usage:
    python scripts/run_p0.py --config configs/p0.yaml [--sample-index N] [--run-id ID]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from nusc_inject import nusc_io, pipeline
from nusc_inject.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p0.yaml"))
    ap.add_argument("--sample-index", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, ROOT)
    if args.sample_index is not None:
        cfg["dataset"]["sample_index"] = args.sample_index
    if args.run_id:
        cfg["run_id"] = args.run_id

    out_dir = ROOT / cfg.get("output_dir", "data/outputs/p0") / cfg["run_id"]
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "logs" / "run.log", mode="w"),
                  logging.StreamHandler()],
    )

    meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])
    objects = [{"model_cfg": cfg["model"], "placement_cfg": cfg["placement"]}]
    res = pipeline.run_sample(cfg, meta, int(cfg["dataset"]["sample_index"]),
                              objects, out_dir)

    print("\n=== P0 done:", res["out_dir"])
    print("validation all_pass =", res["report"]["all_pass"])
    return 0 if res["report"]["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
