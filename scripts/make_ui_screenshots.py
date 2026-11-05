#!/usr/bin/env python
"""Capture screenshots of the placement UI for the docs.

The UI is a real page, so its screenshots go stale whenever the layout changes.
This drives headless Firefox through the WebDriver REST API directly (stdlib
only — no selenium, no playwright) and saves the states the docs talk about.

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
        raise SystemExit(f"timeout waiting for {what or script[:60]}")

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

CLICK = """const cv=document.getElementById('cv');
const r=cv.getBoundingClientRect();
const u=arguments[0], v=arguments[1];
cv.dispatchEvent(new MouseEvent('click',{bubbles:true,
  clientX:r.left+u*r.width/cv.width, clientY:r.top+v*r.height/cv.height}));
return true;"""

SET_SLIDER = """const el=document.getElementById(arguments[0]);
el.value=arguments[1];
el.dispatchEvent(new Event('input',{bubbles:true}));
document.getElementById('v'+arguments[0]).textContent=arguments[1];
return true;"""

READY = "return document.getElementById('cv').width > 100 && " \
        "document.getElementById('model').options.length > 0;"
FRAME = "return document.getElementById('cv').width > 100;"
PREVIEW = "return document.getElementById('info').textContent.includes('size(');"


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
        d.wait_js(READY, what="page + meta")
        time.sleep(2.5)                       # let the frame image decode
        d.wait_js(FRAME, what="camera frame")

        d.shot(OUT / "ui_overview.png")       # 首屏：图 + 控件

        # 点路面右侧车道 -> 线框预览。
        # 这个像素落在 ego≈[19.7,-3]：该处 4 m 邻域与 1 m 邻域的地面高度估计
        # 只差 6 mm。换成地面起伏处的点，4 m 邻域 15 百分位会比近处路面低
        # 10–18 cm，目标就会显得陷进路面（见 docs/placement-ui.md 已知限制）。
        d.js(CLICK, [1040, 600])
        d.wait_js(PREVIEW, what="preview after click")
        time.sleep(1.0)
        d.shot(OUT / "ui_preview.png")

        # 调俯仰/滚转 -> 倾斜姿态
        d.js(SET_SLIDER, ["pitch", 18])
        d.js(SET_SLIDER, ["roll", -10])
        time.sleep(1.0)
        d.shot(OUT / "ui_tilted.png")

        # 渲染（低采样，只为截图快）
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
