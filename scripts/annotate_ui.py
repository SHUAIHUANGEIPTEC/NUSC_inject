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


class Workspace:
    """Lazily loaded scene data and sampled meshes, shared across requests."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])
        self._scenes, self._meshes = {}, {}

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

    def preview(self, req: dict) -> dict:
        """Run the real placement maths and project the resulting box."""
        idx, cam = int(req["sample_index"]), req["cam"]
        sc = self.scene(idx)
        model_cfg = {"path": req["model_path"],
                     "target_max_dim_m": float(req["target_max_dim_m"]),
                     "albedo": float(req.get("albedo", 0.6)),
                     "intensity_gain_k": float(req.get("intensity_gain_k", 200.0)),
                     "category": req.get("category")}
        placement_cfg = {
            "strategy": "ego_offset",
            "offset_in_ego": [float(x) for x in req["offset_ego"]],
            "yaw_in_ego_deg": float(req["yaw"]),
            "pitch_in_ego_deg": float(req.get("pitch", 0.0)),
            "roll_in_ego_deg": float(req.get("roll", 0.0)),
            "ground": {"search_radius_m": 4.0, "z_percentile": 15.0},
        }
        mesh = self.mesh(model_cfg["path"])
        cfg = {**self.cfg, "dataset": {**self.cfg["dataset"], "sample_index": idx}}
        gt = anchor.build_gt(cfg, self.meta, sc["pts"], sample=sc["sample"],
                             model_cfg=model_cfg, placement_cfg=placement_cfg,
                             raw_bbox=(mesh["bbox_min"], mesh["bbox_size"]),
                             model_verts=mesh["verts"])

        c = sc["cams"][cam]
        T = np.array(gt["T_model2global"], dtype=float).reshape(4, 4)
        lo, sz = np.array(gt["raw_bbox_min"]), np.array(gt["raw_bbox_size"])
        corners = np.array([lo + np.array([sx, sy, sz2]) * sz
                            for sx in (0, 1) for sy in (0, 1) for sz2 in (0, 1)])
        hom = np.hstack([corners, np.ones((8, 1))])
        pc = (np.linalg.inv(c["T_cam2global"]) @ (T @ hom.T)).T[:, :3]
        proj = pc @ c["K"].T
        pix = proj[:, :2] / np.where(np.abs(proj[:, 2:3]) < 1e-9, 1e-9, proj[:, 2:3])
        return {
            "corners_px": [[float(a), float(b)] for a, b in pix],
            "in_front": [bool(z > 0.1) for z in pc[:, 2]],
            "edges": [list(e) for e in EDGES],
            "gt": {"size": [float(v) for v in gt["size"]],
                   "translation": [float(v) for v in gt["translation"]],
                   "scale": float(gt["scale"]),
                   "ground_z_lidar": float(gt["ground_z_lidar"]),
                   "yaw_in_lidar_deg": float(gt["placement"]["yaw_in_lidar_deg"])},
        }

    def render(self, req: dict, job: dict) -> Path:
        """Run the real pipeline for one placement (called from a worker thread)."""
        idx = int(req["sample_index"])
        model_cfg = {"path": req["model_path"],
                     "target_max_dim_m": float(req["target_max_dim_m"]),
                     "albedo": float(req.get("albedo", 0.6)),
                     "intensity_gain_k": float(req.get("intensity_gain_k", 200.0)),
                     "category": req.get("category")}
        placement_cfg = {"strategy": "ego_offset",
                         "offset_in_ego": [float(x) for x in req["offset_ego"]],
                         "yaw_in_ego_deg": float(req["yaw"]),
                         "pitch_in_ego_deg": float(req.get("pitch", 0.0)),
                         "roll_in_ego_deg": float(req.get("roll", 0.0)),
                         "ground": {"search_radius_m": 4.0, "z_percentile": 15.0}}
        cfg = json.loads(json.dumps(self.cfg))     # deep copy; run_sample mutates it
        cfg["dataset"]["sample_index"] = idx
        cfg["camera_channel"]["cycles_samples"] = int(req.get("samples", 16))
        out = ROOT / "data" / "outputs" / "ui" / f"run_{uuid.uuid4().hex[:6]}"
        res = pipeline.run_sample(cfg, self.meta, idx,
                                  [{"model_cfg": model_cfg, "placement_cfg": placement_cfg}],
                                  out)
        job["log"] = [f"status={res['status']}", f"all_pass={res['report']['all_pass']}",
                      f"out={out}"]
        if res["status"] != "ok":
            raise RuntimeError(f"run_sample returned {res['status']}")
        return out

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
<html><head><meta charset="utf-8"><title>nusc-inject 放置工具</title>
<style>
 body{font:13px/1.5 system-ui,sans-serif;margin:0;background:#1b1d22;color:#d8dae0}
 header{padding:8px 14px;background:#24272e;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 main{display:flex;gap:14px;padding:14px;align-items:flex-start}
 #stage{flex:1 1 auto;min-width:0;position:relative;background:#000;border-radius:4px;overflow:hidden}
 canvas{display:block;width:100%;height:auto;cursor:crosshair}
 aside{flex:0 0 300px;background:#24272e;border-radius:4px;padding:12px}
 label{display:block;margin:9px 0 2px;color:#9aa0ab;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
 select,input[type=number]{width:100%;background:#1b1d22;color:#d8dae0;border:1px solid #3a3f4a;border-radius:3px;padding:4px}
 input[type=range]{width:100%}
 .row{display:flex;gap:8px}.row>*{flex:1}
 button{background:#3b6ea5;color:#fff;border:0;border-radius:3px;padding:7px 14px;cursor:pointer}
 button:disabled{background:#3a3f4a;cursor:default}
 #hint{color:#9aa0ab;font-size:12px}
 #log{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:11px;background:#15171b;padding:8px;border-radius:3px;max-height:150px;overflow:auto}
 #result{position:absolute;left:0;top:0;width:100%;height:100%;object-fit:contain;
         display:none;background:#000;cursor:zoom-out}
</style></head><body>
<header>
  <strong>nusc-inject · 放置工具</strong>
  <span id="hint">在图上点一下选择放置位置，拖滑条调姿态，然后渲染</span>
</header>
<main>
  <div id="stage">
    <canvas id="cv"></canvas>
    <img id="result" alt="render result">
  </div>
  <aside>
    <label>样本 / 相机</label>
    <div class="row"><select id="idx"></select><select id="cam"></select></div>
    <label>模型</label><select id="model"></select>
    <label>类别</label><input id="category" type="text">
    <label>最长边 (m) / albedo</label>
    <div class="row"><input id="dim" type="number" step="0.1" value="3.0">
      <input id="albedo" type="number" step="0.05" value="0.6" min="0" max="1"></div>
    <label>偏航 <span id="vyaw">0</span>°</label><input id="yaw" type="range" min="-180" max="180" step="1" value="0">
    <label>俯仰 <span id="vpitch">0</span>°</label><input id="pitch" type="range" min="-90" max="90" step="1" value="0">
    <label>滚转 <span id="vroll">0</span>°</label><input id="roll" type="range" min="-90" max="90" step="1" value="0">
    <label>渲染采样数</label><input id="samples" type="number" value="16" min="4" max="512">
    <p><button id="render">渲染（Blender）</button></p>
    <div id="info" style="font-size:12px;color:#9aa0ab"></div>
    <div id="log"></div>
  </aside>
</main>
<script>
const $ = id => document.getElementById(id);
let meta = null, state = null, img = new Image();

function draw() {
  const cv = $('cv'), ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  if (!state) return;
  const c = state.corners_px, f = state.in_front, e = state.edges;
  ctx.lineWidth = 2; ctx.strokeStyle = '#ffd24a'; ctx.setLineDash([]);
  for (const [i, j] of e) {
    if (!f[i] || !f[j]) continue;
    ctx.beginPath(); ctx.moveTo(c[i][0], c[i][1]); ctx.lineTo(c[j][0], c[j][1]); ctx.stroke();
  }
  ctx.fillStyle = '#ffd24a';
  for (let i = 0; i < 8; i++) if (f[i]) ctx.fillRect(c[i][0]-3, c[i][1]-3, 6, 6);
}

function body(extra) {
  return Object.assign({
    sample_index: +$('idx').value, cam: $('cam').value, model_path: $('model').value,
    category: $('category').value || null, target_max_dim_m: +$('dim').value,
    albedo: +$('albedo').value, yaw: +$('yaw').value, pitch: +$('pitch').value,
    roll: +$('roll').value, offset_ego: meta.offset_ego
  }, extra || {});
}

async function preview() {
  const r = await fetch('/api/preview', {method:'POST', body: JSON.stringify(body())});
  if (!r.ok) { $('info').textContent = '预览失败: ' + await r.text(); return; }
  state = await r.json();
  const g = state.gt;
  $('info').textContent =
    `size(w,l,h)=[${g.size.map(v=>v.toFixed(2)).join(', ')}]  scale=${g.scale.toFixed(4)}\n`
    + `ego=[${meta.offset_ego.map(v=>v.toFixed(1)).join(', ')}]  ground_z=${g.ground_z_lidar.toFixed(2)}`;
  draw();
}

async function loadFrame() {
  const r = await fetch(`/api/frame?idx=${$('idx').value}&cam=${$('cam').value}`);
  const blob = await r.blob();
  img.onload = () => {
    $('cv').width = img.naturalWidth; $('cv').height = img.naturalHeight;
    if (meta && meta.offset_ego) preview(); else draw();
  };
  img.src = URL.createObjectURL(blob);
}

$('cv').addEventListener('click', async ev => {
  const cv = $('cv'), rect = cv.getBoundingClientRect();
  const u = (ev.clientX - rect.left) * cv.width / rect.width;
  const v = (ev.clientY - rect.top) * cv.height / rect.height;
  const r = await fetch('/api/pick', {method:'POST',
    body: JSON.stringify({sample_index: +$('idx').value, cam: $('cam').value, u, v})});
  if (!r.ok) { $('info').textContent = '选点失败（可能点到了地平线以上）'; return; }
  meta = Object.assign(meta || {}, await r.json());
  preview();
});

for (const id of ['yaw','pitch','roll']) {
  $(id).addEventListener('input', () => { $('v'+id).textContent = $(id).value; preview(); });
}
for (const id of ['dim','albedo','model','category']) {
  $(id).addEventListener('change', preview);
}
$('idx').addEventListener('change', loadFrame);
$('cam').addEventListener('change', loadFrame);

$('render').addEventListener('click', async () => {
  const btn = $('render'); btn.disabled = true;
  $('log').textContent = '提交中…';
  const r = await fetch('/api/render', {method:'POST', body: JSON.stringify(body())});
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
      $('log').textContent += '\n' + (s.error || '');
    }
  }, 1000);
});

$('result').addEventListener('click', () => { $('result').style.display = 'none'; });

(async () => {
  const m = await (await fetch('/api/meta')).json();
  meta = m;
  $('idx').innerHTML = m.samples.map(i => `<option value="${i}">sample ${i}</option>`).join('');
  $('cam').innerHTML = m.cams.map(c => `<option>${c}</option>`).join('');
  $('model').innerHTML = m.models.map(p => `<option value="${p}">${p.split('/').pop()}</option>`).join('');
  $('category').value = m.default_category;
  loadFrame();
})();
</script></body></html>
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
