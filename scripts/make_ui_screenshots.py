#!/usr/bin/env python
"""Capture screenshots of the placement UI for the docs — and smoke-test it.

The UI is a real page, so its screenshots go stale whenever the layout changes.
This drives headless Firefox through the WebDriver REST API directly (stdlib
only — no selenium, no playwright) and saves the states the docs talk about.
Because it dispatches the same pointer / drag events a human would, a failing
step (JS error, dropped preview, missing thumbnail) shows up as a timeout here,
which makes it the UI's end-to-end smoke test as well as its screenshot rig.

    python scripts/make_ui_screenshots.py            # -> docs/summary/figs/ui_*.png

Requires ``geckodriver`` on PATH (it also needs a Firefox install). The script
starts ``annotate_ui.py`` itself on a scratch port.
"""
from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "summary" / "figs"
UI_PORT = 8791
WD_PORT = 4445
WINDOW = (1580, 800)


# ------------------------------------------------------------------ plumbing


def _free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _wait_port(port, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if not _free(port):
            return True
        time.sleep(0.3)
    return False


class Driver:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"
        body = {"capabilities": {"alwaysMatch": {
            "browserName": "firefox",
            "moz:firefoxOptions": {"args": ["-headless"]}}}}
        self.sid = self._post("/session", body)["value"]["sessionId"]
        self._post(f"/session/{self.sid}/window/rect",
                   {"width": WINDOW[0], "height": WINDOW[1]})

    def _post(self, path, body):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())

    def get(self, path):
        return json.loads(urllib.request.urlopen(self.base + path, timeout=120).read())

    def open(self, url):
        self._post(f"/session/{self.sid}/url", {"url": url})

    def js(self, script, args=None):
        return self._post(f"/session/{self.sid}/execute/sync",
                          {"script": script, "args": args or []})["value"]

    def wait_js(self, script, timeout=120, what=""):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.js(script):
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        raise SystemExit(f"timeout waiting for {what or script[:70]}")

    def shot(self, path: Path):
        data = self.get(f"/session/{self.sid}/screenshot")["value"]
        path.write_bytes(base64.b64decode(data))
        print(f"  {path}  {path.stat().st_size // 1024} KB")

    def close(self):
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{self.base}/session/{self.sid}",
                                       method="DELETE"), timeout=30).read()
        except Exception:
            pass


# --------------------------------------------------------------------- steps

# --- canvas interaction (the UI listens for PointerEvent, not MouseEvent) ---

PEV = """const cv=document.getElementById('cv'), r=cv.getBoundingClientRect();
const type=arguments[0], x=arguments[1], y=arguments[2], shift=arguments[3];
cv.dispatchEvent(new PointerEvent(type,{bubbles:true,cancelable:true,
  pointerId:1,pointerType:'mouse',isPrimary:true,button:0,
  buttons:(type==='pointerup'?0:1),
  clientX:r.left+x*r.width/cv.width, clientY:r.top+y*r.height/cv.height,
  shiftKey:!!shift}));
return true;"""

# --- HTML5 drag & drop from the palette onto the canvas ---------------------

DROPOnto = """const el=document.querySelector(arguments[0]), cv=document.getElementById('cv');
const r=cv.getBoundingClientRect(), x=arguments[1], y=arguments[2];
const dt=new DataTransfer();
el.dispatchEvent(new DragEvent('dragstart',{bubbles:true,cancelable:true,dataTransfer:dt}));
const ev={bubbles:true,cancelable:true,dataTransfer:dt,
  clientX:r.left+x*r.width/cv.width, clientY:r.top+y*r.height/cv.height};
cv.dispatchEvent(new DragEvent('dragover',ev));
cv.dispatchEvent(new DragEvent('drop',ev));
return true;"""

READY = "return document.getElementById('cv').width > 100 && " \
        "document.querySelectorAll('#thumbs .thumb').length > 0;"
THUMBS = "return [...document.querySelectorAll('#thumbs .thumb')]" \
         ".every(i => i.complete && i.naturalWidth > 0);"

# centroid of one object's projected box — the place a user actually clicks.
# Clicking the drop point itself is unreliable: that pixel is the bbox-min
# corner, i.e. ON the convex hull, where point-in-polygon is a coin toss.
CENTROID = """const s=window.__ui().state.objects[arguments[0]];
if (!s) return null;
const c=s.corners_px;
return [c.reduce((a,p)=>a+p[0],0)/c.length, c.reduce((a,p)=>a+p[1],0)/c.length];"""


def wait_n(d, n, what, timeout=120):
    d.wait_js(f"return window.__ui().objects.length === {n};", timeout=timeout, what=what)


def centroid(d, i, what):
    p = d.js(CENTROID, [i])
    if not p:
        raise SystemExit(f"no projected box for object {i} ({what})")
    return p


def main():
    if not shutil.which("geckodriver"):
        sys.exit("geckodriver not found on PATH (needs Firefox too)")

    ui = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "annotate_ui.py"),
                           "--port", str(UI_PORT)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wd = subprocess.Popen(["geckodriver", "--port", str(WD_PORT)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not _wait_port(UI_PORT) or not _wait_port(WD_PORT):
            sys.exit("server or geckodriver did not come up")
        OUT.mkdir(parents=True, exist_ok=True)
        d = Driver(WD_PORT)
        d.open(f"http://127.0.0.1:{UI_PORT}/")
        d.wait_js(READY, what="page + palette")
        # first load renders the thumbnails with Blender (one job per model);
        # cached afterwards. Long timeout because this may be a cold cache.
        d.wait_js(THUMBS, timeout=300, what="model thumbnails")
        time.sleep(1.0)
        d.wait_js("return document.getElementById('cv').height > 100;", what="camera frame")

        d.shot(OUT / "ui_overview.png")       # 首屏：素材面板 + 画布 + 控件

        # 拖皮卡缩略图到路面 -> 插入第 1 个目标（1040,600 是实测地面估计准的点）
        d.js(DROPOnto, [".thumb[data-model='data/models/model.glb']", 1040, 600])
        wait_n(d, 1, "pickup inserted")
        time.sleep(0.8)
        d.shot(OUT / "ui_insert.png")

        # 再拖一个护栏 -> 多目标
        d.js(DROPOnto, [".thumb[data-model='data/models/demo_barrier.glb']", 760, 640])
        wait_n(d, 2, "barrier inserted")
        time.sleep(0.8)
        d.shot(OUT / "ui_two.png")

        # 移动模式：把护栏拖到右边（抓取偏移：车头不跳到车身中心）
        cu, cv = centroid(d, 1, "barrier")
        d.js(PEV, ["pointerdown", cu, cv, False])
        for x in range(int(cu) + 40, int(cu) + 221, 45):
            d.js(PEV, ["pointermove", x, cv - 10, False])
            time.sleep(0.25)
        d.js(PEV, ["pointerup", cu + 180, cv - 10, False])
        time.sleep(0.8)
        d.shot(OUT / "ui_move.png")

        # 旋转模式：拖 X 改偏航，Shift+拖 改滚转
        cu, cv = centroid(d, 1, "barrier")
        d.js("document.getElementById('modeRotate').click(); return true;")
        time.sleep(0.2)
        d.js(PEV, ["pointerdown", cu, cv, False])
        for x in range(int(cu) + 20, int(cu) + 111, 30):
            d.js(PEV, ["pointermove", x, cv, False])
            time.sleep(0.2)
        d.js(PEV, ["pointerup", cu + 110, cv, False])
        d.js(PEV, ["pointerdown", cu + 110, cv, True])
        for x in range(int(cu) + 90, int(cu) - 1, -30):
            d.js(PEV, ["pointermove", x, cv, True])
            time.sleep(0.2)
        d.js(PEV, ["pointerup", cu - 20, cv, True])
        time.sleep(0.8)
        d.shot(OUT / "ui_rotate.png")

        # 点皮卡线框 -> 选中（命中测试 = 8 个投影角点取凸包）
        cu, cv = centroid(d, 0, "pickup")
        d.js(PEV, ["pointerdown", cu, cv, False])
        d.js(PEV, ["pointerup", cu, cv, False])
        d.wait_js("return window.__ui().selected === window.__ui().objects[0].id;",
                  what="select pickup")
        time.sleep(0.6)
        d.shot(OUT / "ui_select.png")

        # 点天空 -> 明确拒绝（只有移动模式才选地面点，先切回来）
        d.js("document.getElementById('modeMove').click(); return true;")
        time.sleep(0.2)
        d.js(PEV, ["pointerdown", 1000, 120, False])
        d.js(PEV, ["pointerup", 1000, 120, False])
        d.wait_js("return document.getElementById('info').textContent"
                  ".includes('选点失败');", what="sky rejection")
        time.sleep(0.4)
        d.shot(OUT / "ui_sky.png")

        # 渲染两个目标（低采样，只为截图快）
        d.js("document.getElementById('samples').value=16;"
             "document.getElementById('render').click(); return true;")
        d.wait_js("const r=document.getElementById('result');"
                  "return r && r.style.display==='block' && r.naturalWidth>0;",
                  timeout=300, what="render result")
        time.sleep(0.6)
        d.shot(OUT / "ui_result.png")
        d.close()
    finally:
        for p in (wd, ui):
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
