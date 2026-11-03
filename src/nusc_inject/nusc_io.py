"""Load nuScenes metadata (raw JSON) and LiDAR .pcd.bin files."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class NuScenesMini:
    """Thin raw-JSON reader for v1.0-mini / v1.0-trainval metadata.

    sample_data.json has no ``channel`` key; channel comes from
    calibrated_sensor.json -> sensor.json (verified on v1.0-mini).
    """

    def __init__(self, root: str, version: str = "v1.0-mini"):
        self.root = Path(root)
        self.meta_dir = self.root / version
        for f in ("sample.json", "sample_data.json", "ego_pose.json",
                  "calibrated_sensor.json", "sensor.json", "scene.json",
                  "sample_annotation.json", "instance.json", "category.json"):
            if not (self.meta_dir / f).exists():
                raise FileNotFoundError(f"missing metadata file: {self.meta_dir / f}")
        self.samples = self._load("sample.json")
        self.sample_data = {r["token"]: r for r in self._load("sample_data.json")}
        self.poses = {r["token"]: r for r in self._load("ego_pose.json")}
        self.calib = {r["token"]: r for r in self._load("calibrated_sensor.json")}
        self.sensor = {r["token"]: r for r in self._load("sensor.json")}
        self.scene = {r["token"]: r for r in self._load("scene.json")}
        # 3D annotation boxes (occluder bodies) + instance/category for names
        self.annotations = self._load("sample_annotation.json")
        self.instances = {r["token"]: r for r in self._load("instance.json")}
        self.categories = {r["token"]: r for r in self._load("category.json")}
        self.ann_by_sample: dict = {}
        for a in self.annotations:
            self.ann_by_sample.setdefault(a["sample_token"], []).append(a)
        self._sample_by_token = {s["token"]: s for s in self.samples}
        self.sample_idx_by_token = {s["token"]: i for i, s in enumerate(self.samples)}
        # token -> channel name (e.g. CAM_FRONT, LIDAR_TOP)
        self.cs_to_channel = {
            cs_token: self.sensor[cs["sensor_token"]]["channel"]
            for cs_token, cs in self.calib.items()
        }

    def _load(self, name):
        with open(self.meta_dir / name) as f:
            return json.load(f)

    def sample(self, index: int) -> dict:
        return self.samples[index]

    def sample_by_token(self, token: str) -> dict:
        return self._sample_by_token[token]

    def scene_samples(self, scene: dict) -> list:
        """All keyframe samples of a scene, in temporal order."""
        out, tok = [], scene["first_sample_token"]
        while tok:
            s = self._sample_by_token[tok]
            out.append(s)
            tok = s["next"]
        if len(out) != scene["nbr_samples"]:
            raise RuntimeError(
                f"scene {scene['name']}: walked {len(out)} != "
                f"nbr_samples {scene['nbr_samples']}")
        return out

    def sd_for_sample(self, sample_token: str, channel: str) -> dict:
        """The KEYFRAME sample_data record of ``channel`` for this sample.

        Sweep records also carry sample_token (nearest keyframe), so the
        ``is_key_frame`` filter is mandatory — otherwise iteration order can
        silently return a swept frame with a different timestamp.
        """
        for sd in self.sample_data.values():
            if sd["sample_token"] != sample_token or not sd.get("is_key_frame", False):
                continue
            if self.cs_to_channel[sd["calibrated_sensor_token"]] == channel:
                return sd
        raise KeyError(f"no keyframe sample_data for sample={sample_token} channel={channel}")

    def annotations_for_sample(self, sample_token: str) -> list:
        """All sample_annotation boxes of one sample (global-frame boxes)."""
        return self.ann_by_sample.get(sample_token, [])

    def category_name_of(self, ann: dict) -> str:
        inst = self.instances[ann["instance_token"]]
        return self.categories[inst["category_token"]]["name"]

    def sensor_path(self, sd: dict) -> Path:
        """Absolute path of one sample_data record's file (lidar or camera)."""
        return self.root / sd["filename"]


def load_lidar_bin(path) -> np.ndarray:
    """nuScenes LIDAR_TOP .pcd.bin -> (N,5) float32: x,y,z,intensity,ring.

    Native v1.0-mini layout is 5 floats/point (verified); a (N,4) file is
    also accepted so LIDO-style clouds can be read back.
    """
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5 == 0:
        return raw.reshape(-1, 5)
    if raw.size % 4 == 0:
        pts = raw.reshape(-1, 4)
        ring = np.zeros((pts.shape[0], 1), dtype=np.float32)
        return np.hstack([pts, ring])
    raise ValueError(f"unexpected lidar file size {raw.size} floats: {path}")


def save_lidar_bin(path, pts: np.ndarray) -> None:
    """Write (N,4) or (N,5) float32; stored as-is."""
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] not in (4, 5):
        raise ValueError(f"expected (N,4) or (N,5), got {arr.shape}")
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    arr.tofile(path)
