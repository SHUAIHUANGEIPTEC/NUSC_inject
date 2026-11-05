#!/usr/bin/env python
"""Interactive placement UI: click a spot in the camera frame, tune the pose, render.

    python scripts/annotate_ui.py                 # -> http://127.0.0.1:8765
    python scripts/annotate_ui.py --port 9000 --open

Why hand-rolled instead of gradio/streamlit: this repo keeps its dependency
surface pinned and small, and the UI needs exactly one image, one click, four
sliders and one button. `http.server` + a single page covers that with zero new
dependencies.

How it works:

- **placing**: a click gives a pixel; the ray through it is intersected with the
  estimated ground plane in the LIDAR frame, so the resulting ego-frame offset
  means the same thing a config file would (object dropped on the ground).
- **preview**: the wireframe is built by calling ``anchor.build_gt`` — the very
  function the real run uses — so what you see is what will be rendered. No
  second copy of the placement maths to drift out of sync.
- **render**: the real pipeline (Blender included) runs in a background thread;
  Blender takes tens of seconds, so the page polls instead of blocking.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nusc_inject import anchor, camera_channel, mesh_points, nusc_io, pipeline  # noqa: E402
from nusc_inject import transforms as tf  # noqa: E402
from nusc_inject.config import load_config  # noqa: E402

# corner index = 4*sx + 2*sy + sz, so edges differ in exactly one bit
EDGES = ((0, 1), (2, 3), (4, 5), (6, 7),      # sz
         (0, 2), (1, 3), (4, 6), (5, 7),      # sy
         (0, 4), (1, 5), (2, 6), (3, 7))      # sx


def _thumb_job(model_path: str, bmin, bsize, verts, out_dir: str) -> dict:
    """Build a Blender job that renders one model as a 256px RGBA thumbnail.

    One job per model on purpose: the render path loads every object into a
    shared scene and passes them all to every camera (blender_render.py
    render_one), so N models in one job would be photographed together.
    """
    bmin = np.asarray(bmin, dtype=float)
    bsize = np.asarray(bsize, dtype=float)
    s = 1.0 / float(bsize.max())
    T = np.eye(4)
    T[:3, :3] *= s
    T[:3, 3] = -s * (bmin + bsize / 2.0)           # model centre -> the origin

    # camera: 3/4 view looking at the origin. Distance is solved so the
    # projected silhouette fills ~80% of the frame — a fixed multiple of the
    # bounding radius under- or over-shoots depending on the shape.
    az, el = math.radians(30.0), math.radians(15.0)
    fwd = np.array([math.cos(el) * math.cos(az),
                    math.cos(el) * math.sin(az), math.sin(el)])
    z = -fwd                                       # camera z = forward = at origin
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                             # camera y = down
    R = np.stack([x, y, z], axis=1)
    f = 256 * 0.6
    K = np.array([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]])
    local = (np.asarray(verts, dtype=float) - (bmin + bsize / 2.0)) * s

    r = float(np.linalg.norm(bsize)) / 2.0 * s
    target = 0.80 * 256

    def span_at(d: float) -> float:
        """Silhouette span in px at camera distance d (+inf when too close)."""
        pc = (R.T @ local.T).T + np.array([0.0, 0.0, d])
        if (pc[:, 2] <= 1e-6).any():                # camera inside the model
            return float("inf")
        uv = (K @ pc.T).T
        uv = uv[:, :2] / pc[:, 2:3]
        return max(float(uv[:, 0].ptp()), float(uv[:, 1].ptp()))

    # span decreases monotonically with d, so bisect rather than iterate:
    # d *= span/target oscillates under perspective (span is not proportional
    # to 1/d at this FOV) — measured 126 -> 312 -> 137 -> 295 -> ...
    d_hi = max(4.0 * r, 1.0)
    while span_at(d_hi) > target and d_hi < 1e4:
        d_hi *= 1.5
    d_lo = 1e-3
    for _ in range(60):
        mid = 0.5 * (d_lo + d_hi)
        if span_at(mid) > target:
            d_lo = mid
        else:
            d_hi = mid
    d = 0.5 * (d_lo + d_hi)
    T_c2g = np.eye(4)
    T_c2g[:3, :3] = R
    T_c2g[:3, 3] = fwd * d

    return {"mode": "render",
            "objects": [{"model_path": model_path,
                         "T_model2global": [float(v) for v in T.reshape(-1)]}],
            "cams": [{"name": "thumb", "K": K.tolist(),
                      "T_cam2global": [float(v) for v in T_c2g.reshape(-1)],
                      "width": 256, "height": 256}],
            "engine": "CYCLES", "cycles_device": "CPU", "cycles_samples": 16,
            "output_dir": out_dir}


class Workspace:
    """Lazily loaded scene data and sampled meshes, shared across requests."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])
        self._scenes, self._meshes = {}, {}
        self._thumb_lock = threading.Lock()   # one Blender at a time

    # ---------------------------------------------------------------- scene
    def scene(self, idx: int) -> dict:
        if idx in self._scenes:
            return self._scenes[idx]
        sample = self.meta.sample(int(idx))
        lidar_sd = self.meta.sd_for_sample(sample["token"], self.cfg["dataset"]["lidar_channel"])
        cs = self.meta.calib[lidar_sd["calibrated_sensor_token"]]
        pose = self.meta.poses[lidar_sd["ego_pose_token"]]
        T_l2e = tf.make_T(cs["rotation"], cs["translation"])
        T_e2g = tf.make_T(pose["rotation"], pose["translation"])
        pts = nusc_io.load_lidar_bin(self.meta.sensor_path(lidar_sd))

        cams = {}
        for ch in self.cfg["dataset"]["cam_channels"]:
            try:
                sd = self.meta.sd_for_sample(sample["token"], ch)
            except KeyError:
                continue
            cc = self.meta.calib[sd["calibrated_sensor_token"]]
            cams[ch] = {
                "K": np.array(cc["camera_intrinsic"], dtype=float),
                "T_cam2global": np.linalg.inv(camera_channel.global2cam(sd, self.meta)),
                "width": int(sd["width"]), "height": int(sd["height"]),
                "path": self.meta.sensor_path(sd),
            }
        sc = {"sample": sample, "pts": pts, "T_l2e": T_l2e, "T_e2g": T_e2g,
              "T_l2g": T_e2g @ T_l2e, "cams": cams,
              "ground_z": float(np.percentile(pts[:, 2], 15.0))}
        self._scenes[idx] = sc
        return sc

    def mesh(self, path: str):
        # one cache entry per FILE, not per spelling: "a/../a/x.glb" and "a/x.glb"
        # must not sample twice (and must both satisfy the drift guard).
        path = os.path.normpath(path)
        if path not in self._meshes:
            pts, normals, bmin, bsize, _, verts = mesh_points.sample_mesh(
                path, int(self.cfg["mesh_points"]["n_dense"]))
            self._meshes[path] = {"pts": pts, "normals": normals,
                                  "bbox_min": bmin, "bbox_size": bsize,
                                  "verts": verts}
        return self._meshes[path]

    # ------------------------------------------------------------ placement
    def _ground_at(self, sc, x, y) -> float:
        radius = 4.0
        d2 = (sc["pts"][:, 0] - x) ** 2 + (sc["pts"][:, 1] - y) ** 2
        near = sc["pts"][d2 <= radius * radius]
        if len(near) < 10:
            near = sc["pts"]
        return float(np.percentile(near[:, 2], 15.0))

    def pick(self, idx: int, cam: str, u: float, v: float) -> dict:
        """Pixel -> ego-frame offset, by hitting the estimated ground plane."""
        sc = self.scene(idx)
        c = sc["cams"][cam]
        R_g2l = np.linalg.inv(sc["T_l2g"])[:3, :3]
        T_g2l = np.linalg.inv(sc["T_l2g"])

        # ray in the camera frame (OpenCV: z forward), then lifted to global
        d_cam = np.linalg.inv(c["K"]) @ np.array([u, v, 1.0])
        d_g = c["T_cam2global"][:3, :3] @ d_cam
        origin_l = R_g2l @ c["T_cam2global"][:3, 3] + T_g2l[:3, 3]
        dir_l = R_g2l @ d_g

        gz = sc["ground_z"]
        p_l = None
        for _ in range(3):                      # ground height depends on x/y
            if abs(dir_l[2]) < 1e-6:
                break
            t = (gz - origin_l[2]) / dir_l[2]
            if t <= 0:
                break
            p_l = origin_l + t * dir_l
            gz = self._ground_at(sc, float(p_l[0]), float(p_l[1]))
        if p_l is None:
            raise ValueError("该像素指向天空（与地面无交点），请点路面")
        p_ego = tf.transform_pts(sc["T_l2e"], p_l[None, :])[0]
        return {"offset_ego": [float(p_ego[0]), float(p_ego[1]), 0.0],
                "ground_z": float(gz)}

    # ------------------------------------------------- per-object placement
    @staticmethod
    def _obj_cfg(o: dict):
        """One UI object -> the (model_cfg, placement_cfg) pair run_sample wants."""
        model_cfg = {"path": o["model_path"],
                     "target_max_dim_m": float(o.get("target_max_dim_m", 3.0)),
                     "albedo": float(o.get("albedo", 0.6)),
                     "intensity_gain_k": float(o.get("intensity_gain_k", 200.0)),
                     "category": o.get("category")}
        placement_cfg = {"strategy": "ego_offset",
                         "offset_in_ego": [float(x) for x in o["offset_ego"]],
                         "yaw_in_ego_deg": float(o.get("yaw", 0.0)),
                         "pitch_in_ego_deg": float(o.get("pitch", 0.0)),
                         "roll_in_ego_deg": float(o.get("roll", 0.0)),
                         "ground": {"search_radius_m": 4.0, "z_percentile": 15.0}}
        return model_cfg, placement_cfg

    def _one_preview(self, idx: int, sc: dict, cam: dict, o: dict) -> dict:
        """Real placement maths for one object + its projected box corners."""
        model_cfg, placement_cfg = self._obj_cfg(o)
        mesh = self.mesh(model_cfg["path"])
        cfg = {**self.cfg, "dataset": {**self.cfg["dataset"], "sample_index": idx}}
        gt = anchor.build_gt(cfg, self.meta, sc["pts"], sample=sc["sample"],
                             model_cfg=model_cfg, placement_cfg=placement_cfg,
                             raw_bbox=(mesh["bbox_min"], mesh["bbox_size"]),
                             model_verts=mesh["verts"])

        T = np.array(gt["T_model2global"], dtype=float).reshape(4, 4)
        lo, sz = np.array(gt["raw_bbox_min"]), np.array(gt["raw_bbox_size"])
        corners = np.array([lo + np.array([sx, sy, sz2]) * sz
                            for sx in (0, 1) for sy in (0, 1) for sz2 in (0, 1)])
        hom = np.hstack([corners, np.ones((8, 1))])
        pc = (np.linalg.inv(cam["T_cam2global"]) @ (T @ hom.T)).T[:, :3]
        proj = pc @ cam["K"].T
        pix = proj[:, :2] / np.where(np.abs(proj[:, 2:3]) < 1e-9, 1e-9, proj[:, 2:3])
        return {
            "id": o.get("id"),
            "corners_px": [[float(a), float(b)] for a, b in pix],
            "in_front": [bool(z > 0.1) for z in pc[:, 2]],
            "edges": [list(e) for e in EDGES],
            "gt": {"size": [float(v) for v in gt["size"]],
                   "translation": [float(v) for v in gt["translation"]],
                   "scale": float(gt["scale"]),
                   "ground_z_lidar": float(gt["ground_z_lidar"]),
                   "yaw_in_lidar_deg": float(gt["placement"]["yaw_in_lidar_deg"])},
        }

    def preview(self, req: dict) -> dict:
        """Real placement maths + projected box corners, for every object."""
        idx, cam = int(req["sample_index"]), req["cam"]
        sc = self.scene(idx)
        c = sc["cams"][cam]
        return {"objects": [self._one_preview(idx, sc, c, o) for o in req["objects"]]}

    def render(self, req: dict, job: dict) -> Path:
        """Run the real pipeline for every object (called from a worker thread)."""
        idx = int(req["sample_index"])
        objs = req["objects"]
        ids = [o.get("id") for o in objs]
        specs = [{"model_cfg": m, "placement_cfg": p}
                 for m, p in (self._obj_cfg(o) for o in objs)]
        cfg = json.loads(json.dumps(self.cfg))     # deep copy; run_sample mutates it
        cfg["dataset"]["sample_index"] = idx
        cfg["camera_channel"]["cycles_samples"] = int(req.get("samples", 16))
        out = ROOT / "data" / "outputs" / "ui" / f"run_{uuid.uuid4().hex[:6]}"
        # drop_invisible=True: one object hidden behind real geometry must not
        # fail the whole render. dropped holds REQUEST INDICES, not client ids.
        res = pipeline.run_sample(cfg, self.meta, idx, specs, out, drop_invisible=True)
        dropped = [ids[i] for i in res.get("dropped", [])]
        job["log"] = [f"status={res['status']}",
                      f"all_pass={(res.get('report') or {}).get('all_pass')}",
                      f"objects={len(ids)} dropped={dropped or '-'}",
                      f"out={out}"]
        if res["status"] != "ok":
            raise RuntimeError(f"run_sample returned {res['status']}")
        return out

    # ------------------------------------------------------------- thumbnails
    def thumb(self, model_path: str) -> Path:
        """Render (and cache) a 256px RGBA preview of one model."""
        key = Path(model_path).stem
        dir_ = ROOT / "data" / "outputs" / "ui" / "thumbs" / key
        png = dir_ / "thumb.png"
        src = Path(model_path) if os.path.isabs(model_path) else ROOT / model_path
        if png.exists() and png.stat().st_mtime > src.stat().st_mtime:
            return png
        m = self.mesh(model_path)
        job = _thumb_job(model_path, m["bbox_min"], m["bbox_size"], m["verts"], str(dir_))
        dir_.mkdir(parents=True, exist_ok=True)
        with self._thumb_lock:
            camera_channel.run_blender(self.cfg, job, dir_)
        return png

    def meta_info(self) -> dict:
        models = sorted(str(p.relative_to(ROOT))
                        for p in (ROOT / "data" / "models").glob("*.glb"))
        # put the default model first: the rest are demo / axis fixtures
        models.sort(key=lambda p: (not p.endswith("/model.glb"), p))
        n = len(self.meta.samples)
        step = max(1, n // 200)
        return {"samples": list(range(0, n, step)),
                "cams": list(self.cfg["dataset"]["cam_channels"]),
                "models": models,
                "categories": sorted({c["name"] for c in self.meta.categories.values()}),
                "default_category": (self.cfg.get("annotation") or {}).get(
                    "category_name", "movable_object.anomaly")}


class Jobs:
    """Background render jobs (Blender takes tens of seconds)."""

    def __init__(self):
        self._jobs: dict = {}
        self._lock = threading.Lock()

    def start(self, fn) -> str:
        jid = uuid.uuid4().hex[:8]
        with self._lock:
            self._jobs[jid] = {"status": "running", "log": [], "out_dir": None}
        def run():
            try:
                out = fn(self._jobs[jid])
                with self._lock:
                    self._jobs[jid].update(status="done", out_dir=str(out))
            except Exception as exc:                       # surfaced to the page
                with self._lock:
                    self._jobs[jid].update(status="error", error=repr(exc))
        threading.Thread(target=run, daemon=True).start()
        return jid

    def get(self, jid: str) -> dict:
        with self._lock:
            return dict(self._jobs.get(jid, {"status": "unknown"}))


# --------------------------------------------------------------------- page

PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>nusc-inject · 放置工具</title>
<style>
 body{font:13px/1.5 system-ui,sans-serif;margin:0;background:#1b1d22;color:#d8dae0}
 header{padding:8px 14px;background:#24272e;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 main{display:flex;gap:12px;padding:12px;align-items:flex-start}
 #palette{flex:0 0 152px;background:#24272e;border-radius:4px;padding:10px;max-height:calc(100vh - 92px);overflow:auto}
 #stage{flex:1 1 auto;min-width:0;position:relative;background:#000;border-radius:4px;overflow:hidden}
 canvas{display:block;width:100%;height:auto;cursor:grab;touch-action:none}
 aside{flex:0 0 300px;background:#24272e;border-radius:4px;padding:12px}
 label{display:block;margin:9px 0 2px;color:#9aa0ab;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
 select,input[type=number],input[type=text]{width:100%;background:#1b1d22;color:#d8dae0;border:1px solid #3a3f4a;border-radius:3px;padding:4px}
 input[type=range]{width:100%}
 input:disabled{opacity:.4}
 .row{display:flex;gap:8px}
 button{background:#2f6fb8;color:#fff;border:0;border-radius:3px;padding:6px 12px;cursor:pointer}
 button:disabled{background:#3a3f4a;cursor:default}
 .mode{background:#1b1d22;color:#d8dae0;border:1px solid #3a3f4a}
 .mode.on{background:#2f6fb8;border-color:#2f6fb8;color:#fff}
 #hint{color:#9aa0ab;font-size:12px}
 .small{font-size:12px;color:#9aa0ab}
 #log{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:11px;background:#15171b;padding:8px;border-radius:3px;max-height:150px;overflow:auto}
 #result{position:absolute;left:0;top:0;width:100%;height:100%;object-fit:contain;display:none;background:#000;cursor:zoom-out}
 .thumb{display:block;width:128px;height:92px;object-fit:contain;background:#111;border:1px solid #3a3f4a;border-radius:3px;margin:5px 0;cursor:grab}
 .thumb.sel{border-color:#ffd24a}
 .chip{display:inline-block;padding:3px 7px;margin:2px;background:#1b1d22;border:1px solid #3a3f4a;border-radius:12px;cursor:grab;font-size:11px}
 .chip.sel{border-color:#ffd24a;color:#ffd24a}
</style>
</head>
<body>
<header>
  <strong>nusc-inject · 放置工具</strong>
  <button class="mode on" id="modeMove">移动</button>
  <button class="mode" id="modeRotate">旋转</button>
  <button id="del" disabled>删除目标</button>
  <span id="hint">拖左侧缩略图 / 类别到画布插入；移动模式拖动改位置，旋转模式拖动改姿态</span>
</header>
<main>
  <aside id="palette">
    <label>模型（拖入画布）</label>
    <div id="thumbs"></div>
    <label>类别（拖入画布 / 拖到目标上）</label>
    <div id="cats"></div>
  </aside>
  <div id="stage">
    <canvas id="cv"></canvas>
    <img id="result" alt="render result">
  </div>
  <aside>
    <label>样本 / 相机</label>
    <div class="row"><select id="idx"></select><select id="cam"></select></div>
    <label>当前目标</label>
    <div id="selinfo" class="small">（未选中——先拖一个目标进来，或点画布上的线框）</div>
    <label>类别</label><input id="category" type="text" disabled>
    <label>最长边 (m) / albedo</label>
    <div class="row"><input id="dim" type="number" step="0.1" value="3.0" disabled>
      <input id="albedo" type="number" step="0.05" value="0.6" min="0" max="1" disabled></div>
    <label>偏航 <span id="vyaw">0</span>°</label><input id="yaw" type="range" min="-180" max="180" step="1" value="0" disabled>
    <label>俯仰 <span id="vpitch">0</span>°</label><input id="pitch" type="range" min="-90" max="90" step="1" value="0" disabled>
    <label>滚转 <span id="vroll">0</span>°</label><input id="roll" type="range" min="-90" max="90" step="1" value="0" disabled>
    <label>渲染采样数</label><input id="samples" type="number" value="16" min="4" max="512">
    <p><button id="render">渲染（Blender）</button></p>
    <div id="info" class="small"></div>
    <div id="log"></div>
  </aside>
</main>
<script>
const $ = id => document.getElementById(id);
let meta = null, img = new Image();
let objects = [], selected = null, mode = 'move';
let state = {objects: []};
let pending = false, drag = null, lastPt = null;

const uid = () => 'o' + Math.random().toString(36).slice(2, 9);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const wrap = d => { while (d > 180) d -= 360; while (d <= -180) d += 360; return d; };
const byId = id => objects.find(o => o.id === id);

// ------------------------------------------------------- canvas geometry
function evPos(ev) {
  const cv = $('cv'), r = cv.getBoundingClientRect();
  return [(ev.clientX - r.left) * cv.width / r.width,
          (ev.clientY - r.top) * cv.height / r.height];
}
function hull(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) {
    while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop();
    lo.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop();
    up.push(q);
  }
  lo.pop(); up.pop();
  return lo.concat(up);
}
function inPoly(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if (((yi > pt[1]) !== (yj > pt[1])) &&
        (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}
function hitTest(u, v) {
  for (let i = state.objects.length - 1; i >= 0; i--) {
    const o = state.objects[i];
    if (!byId(o.id)) continue;
    const pts = [];
    for (let k = 0; k < 8; k++) if (o.in_front[k]) pts.push(o.corners_px[k]);
    if (pts.length < 3) continue;
    if (inPoly([u, v], hull(pts))) return o.id;
  }
  return null;
}

// --------------------------------------------------------------- drawing
function draw() {
  const cv = $('cv'), ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  for (const o of state.objects) {
    const on = (o.id === selected);
    ctx.lineWidth = on ? 3 : 1.5;
    ctx.strokeStyle = on ? '#ffd24a' : 'rgba(255,210,74,.42)';
    for (const [i, j] of o.edges) {
      if (!o.in_front[i] || !o.in_front[j]) continue;
      ctx.beginPath();
      ctx.moveTo(o.corners_px[i][0], o.corners_px[i][1]);
      ctx.lineTo(o.corners_px[j][0], o.corners_px[j][1]);
      ctx.stroke();
    }
    if (on) {
      ctx.fillStyle = '#ffd24a';
      for (let k = 0; k < 8; k++) if (o.in_front[k])
        ctx.fillRect(o.corners_px[k][0] - 3, o.corners_px[k][1] - 3, 6, 6);
    }
  }
}

// --------------------------------------------------------------- server
function body(extra) {
  return Object.assign({
    sample_index: +$('idx').value, cam: $('cam').value,
    objects: objects.map(o => ({...o}))
  }, extra || {});
}
async function preview() {
  if (!objects.length) { state = {objects: []}; info(); draw(); return; }
  const r = await fetch('/api/preview', {method: 'POST', body: JSON.stringify(body())});
  if (!r.ok) { $('info').textContent = '预览失败: ' + await r.text(); return; }
  state = await r.json();
  info();
  draw();
}
function schedule() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(async () => {
    pending = false;
    if (drag && lastPt && mode === 'move' && drag.grab) {
      const o = byId(drag.target);
      if (o) {
        try {
          const p = await pickAt(lastPt[0], lastPt[1]);
          o.offset_ego = [p.offset_ego[0] - drag.grab[0],
                          p.offset_ego[1] - drag.grab[1], 0.0];
        } catch (e) { /* off the ground plane: keep the previous position */ }
      }
    }
    preview();
  });
}
async function pickAt(u, v) {
  const r = await fetch('/api/pick', {method: 'POST',
    body: JSON.stringify({sample_index: +$('idx').value, cam: $('cam').value, u, v})});
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
function info() {
  const o = byId(selected);
  $('selinfo').textContent = o
    ? `${o.model_path.split('/').pop()} · ${o.category || '（默认类别）'} · ` +
      `yaw ${o.yaw}° pitch ${o.pitch}° roll ${o.roll}°`
    : '（未选中——先拖一个目标进来，或点画布上的线框）';
  const cur = state.objects.find(x => x.id === selected) || state.objects[0];
  if (!cur) { $('info').textContent = `目标数 ${objects.length}`; return; }
  const g = cur.gt;
  $('info').textContent =
    `目标数 ${objects.length}\n` +
    `size(w,l,h)=[${g.size.map(v => v.toFixed(2)).join(', ')}]  scale=${g.scale.toFixed(4)}\n` +
    `ego=[${o ? o.offset_ego.map(v => v.toFixed(1)).join(', ') : '-'}]  ` +
    `ground_z=${g.ground_z_lidar.toFixed(2)}`;
}
async function loadFrame() {
  const r = await fetch(`/api/frame?idx=${$('idx').value}&cam=${$('cam').value}`);
  const blob = await r.blob();
  img.onload = () => {
    $('cv').width = img.naturalWidth;
    $('cv').height = img.naturalHeight;
    preview();                       // 多目标：换相机后全部重预览
  };
  img.src = URL.createObjectURL(blob);
}

// -------------------------------------------------------------- controls
function syncControls() {
  const o = byId(selected);
  for (const id of ['category', 'dim', 'albedo', 'yaw', 'pitch', 'roll']) $(id).disabled = !o;
  $('del').disabled = !o;
  if (!o) return;
  $('category').value = o.category || '';
  $('dim').value = o.target_max_dim_m;
  $('albedo').value = o.albedo;
  $('yaw').value = o.yaw; $('pitch').value = o.pitch; $('roll').value = o.roll;
  $('vyaw').textContent = o.yaw; $('vpitch').textContent = o.pitch; $('vroll').textContent = o.roll;
}
function readControls() {
  const o = byId(selected);
  if (!o) return;
  o.category = $('category').value || null;
  o.target_max_dim_m = +$('dim').value;
  o.albedo = +$('albedo').value;
  o.yaw = +$('yaw').value; o.pitch = +$('pitch').value; o.roll = +$('roll').value;
}
function select(id) {
  selected = id;
  syncControls();
  draw();
}
function setMode(m) {
  mode = m;
  $('modeMove').classList.toggle('on', m === 'move');
  $('modeRotate').classList.toggle('on', m === 'rotate');
  $('cv').style.cursor = m === 'move' ? 'grab' : 'crosshair';
  $('hint').textContent = m === 'move'
    ? '移动模式：拖动改位置；点空白处把选中目标放到那里'
    : '旋转模式：拖 X 改偏航、拖 Y 改俯仰、按住 Shift 拖改滚转';
}

// ----------------------------------------------------------- pointer drag
$('cv').addEventListener('pointerdown', async ev => {
  const [u, v] = evPos(ev);
  const hit = hitTest(u, v);
  if (hit) select(hit);
  drag = {u0: u, v0: v, moved: 0, target: hit || selected, grab: null};
  lastPt = [u, v];
  try { $('cv').setPointerCapture(ev.pointerId); } catch (e) { /* synthetic events */ }
  if (mode === 'move' && drag.target) {
    try {
      const p = await pickAt(u, v), o = byId(drag.target);
      if (o) drag.grab = [p.offset_ego[0] - o.offset_ego[0],
                          p.offset_ego[1] - o.offset_ego[1]];
    } catch (e) {
      drag.target = null;
      $('info').textContent = '选点失败（可能点到了地平线以上）';
    }
  }
});
$('cv').addEventListener('pointermove', ev => {
  if (!drag) return;
  const [u, v] = evPos(ev);
  const dx = u - drag.u0, dy = v - drag.v0;
  drag.moved = Math.max(drag.moved, Math.hypot(dx, dy));
  lastPt = [u, v];
  if (mode === 'rotate') {
    const o = byId(drag.target);
    if (o) {
      const s = ev.shiftKey ? 0.3 : 0.4;
      if (ev.shiftKey) o.roll = clamp(o.roll + dx * s, -90, 90);
      else { o.yaw = wrap(o.yaw + dx * s); o.pitch = clamp(o.pitch - dy * s, -90, 90); }
      drag.u0 = u; drag.v0 = v;
      syncControls();
    }
  }
  schedule();
});
$('cv').addEventListener('pointerup', async ev => {
  if (!drag) return;
  const d = drag; drag = null;
  if (d.moved >= 3) return;                       // it was a drag, not a click
  const [u, v] = evPos(ev);
  const hit = hitTest(u, v);
  if (hit) { select(hit); return; }
  if (selected && mode === 'move') {
    try {
      const p = await pickAt(u, v), o = byId(selected);
      if (o) { o.offset_ego = p.offset_ego; schedule(); }
    } catch (e) {
      $('info').textContent = '选点失败（可能点到了地平线以上）';
    }
  }
});

// --------------------------------------------------------- palette drop
function buildPalette(m) {
  $('thumbs').innerHTML = m.models.map(p =>
    `<img class="thumb" draggable="true" data-model="${p}" alt="${p}" title="${p}" ` +
    `src="/api/thumb?model=${encodeURIComponent(p)}">`).join('');
  $('cats').innerHTML = m.categories.map(c =>
    `<span class="chip" draggable="true" data-category="${c}">${c}</span>`).join('');
  for (const el of document.querySelectorAll('#palette [draggable]')) {
    el.addEventListener('dragstart', ev => {
      ev.dataTransfer.setData('text/plain', JSON.stringify({
        model: el.dataset.model || null, category: el.dataset.category || null}));
    });
  }
}
$('cv').addEventListener('dragover', ev => ev.preventDefault());
$('cv').addEventListener('drop', async ev => {
  ev.preventDefault();
  const [u, v] = evPos(ev);
  let data = {};
  try { data = JSON.parse(ev.dataTransfer.getData('text/plain')); } catch (e) {}
  const hit = hitTest(u, v);
  if (hit && !data.model && data.category) {      // 类别 chip 落到已有目标 = 改类别
    const o = byId(hit); o.category = data.category;
    select(hit); schedule(); return;
  }
  let p;
  try { p = await pickAt(u, v); }
  catch (e) { $('info').textContent = '选点失败（可能点到了地平线以上）'; return; }
  if (hit && data.model) {                        // 模型缩略图落到已有目标 = 换模型
    const o = byId(hit); o.model_path = data.model;
    select(hit); schedule(); return;
  }
  const o = {
    id: uid(),
    model_path: data.model || meta.models[0],
    category: data.category || $('category').value || meta.default_category,
    target_max_dim_m: +$('dim').value || 3.0,
    albedo: +$('albedo').value || 0.6,
    yaw: 0, pitch: 0, roll: 0,
    offset_ego: p.offset_ego,
  };
  objects.push(o);
  select(o.id);
  schedule();
});

// ------------------------------------------------------------- listeners
for (const id of ['yaw', 'pitch', 'roll']) {
  $(id).addEventListener('input', () => {
    $('v' + id).textContent = $(id).value;
    readControls(); schedule();
  });
}
for (const id of ['dim', 'albedo', 'category']) {
  $(id).addEventListener('change', () => { readControls(); schedule(); });
}
$('idx').addEventListener('change', loadFrame);
$('cam').addEventListener('change', loadFrame);
$('modeMove').addEventListener('click', () => setMode('move'));
$('modeRotate').addEventListener('click', () => setMode('rotate'));
$('del').addEventListener('click', () => {
  objects = objects.filter(o => o.id !== selected);
  selected = null; syncControls(); schedule();
});
$('render').addEventListener('click', async () => {
  if (!objects.length) { $('log').textContent = '先拖一个目标进来'; return; }
  const btn = $('render'); btn.disabled = true;
  $('log').textContent = '提交中…';
  const r = await fetch('/api/render', {method: 'POST', body: JSON.stringify(body())});
  const {job} = await r.json();
  const poll = setInterval(async () => {
    const s = await (await fetch('/api/render?id=' + job)).json();
    $('log').textContent = (s.log || []).slice(-12).join('\n') || s.status;
    if (s.status === 'done') {
      clearInterval(poll); btn.disabled = false;
      $('result').src = `/api/result?id=${job}&cam=${$('cam').value}&t=${Date.now()}`;
      $('result').style.display = 'block';
    } else if (s.status === 'error') {
      clearInterval(poll); btn.disabled = false;
    }
  }, 1000);
});
$('result').addEventListener('click', () => { $('result').style.display = 'none'; });

(async () => {
  const m = await (await fetch('/api/meta')).json();
  meta = m;
  $('idx').innerHTML = m.samples.map(i => `<option value="${i}">sample ${i}</option>`).join('');
  $('cam').innerHTML = m.cams.map(c => `<option>${c}</option>`).join('');
  $('category').value = m.default_category;
  buildPalette(m);
  setMode('move');
  window.__ui = () => ({objects, selected, mode, state});   // automation hook
  loadFrame();
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    workspace: Workspace = None
    jobs: Jobs = None

    def log_message(self, *a):        # keep the console clean
        pass

    # ------------------------------------------------------------------ util
    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    # ------------------------------------------------------------------- GET
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/meta":
            return self._json(self.workspace.meta_info())
        if u.path == "/api/frame":
            sc = self.workspace.scene(int(q["idx"][0]))
            c = sc["cams"][q["cam"][0]]
            return self._send(200, Path(c["path"]).read_bytes(), "image/jpeg")
        if u.path == "/api/thumb":
            return self._send(200, self.workspace.thumb(q["model"][0]).read_bytes(),
                              "image/png")
        if u.path == "/api/render":
            return self._json(self.jobs.get(q["id"][0]))
        if u.path == "/api/result":
            job = self.jobs.get(q["id"][0])
            if job.get("status") != "done":
                return self._send(404, b"not ready", "text/plain")
            p = Path(job["out_dir"]) / "composite" / f"{q['cam'][0]}.jpg"
            if not p.exists():
                return self._send(404, b"no such camera in output", "text/plain")
            return self._send(200, p.read_bytes(), "image/jpeg")
        return self._send(404, b"not found", "text/plain")

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/pick":
                return self._json(self.workspace.pick(
                    int(req["sample_index"]), req["cam"], float(req["u"]), float(req["v"])))
            if self.path == "/api/preview":
                return self._json(self.workspace.preview(req))
            if self.path == "/api/render":
                return self._json({"job": self.jobs.start(
                    lambda job: self.workspace.render(req, job))})
        except Exception as exc:
            return self._send(500, repr(exc).encode(), "text/plain")
        return self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "batch.yaml"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args()

    ws = Workspace(load_config(args.config, ROOT))
    Handler.workspace, Handler.jobs = ws, Jobs()
    url = f"http://{args.host}:{args.port}"
    print(f"放置工具: {url}   （Ctrl-C 退出）")
    if args.open:
        webbrowser.open(url)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
