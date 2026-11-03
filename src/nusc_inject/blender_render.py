"""Runs INSIDE Blender's bundled Python (bpy + numpy + stdlib only).

Invoked as:
    blender --background --python blender_render.py -- --job job.json

job.json schema (mode="render"):
{
  "mode": "render",
  "model_path": "....glb",
  "T_model2global": [16 floats, row-major 4x4],   # full (scale+norm+pose) single truth
  "cams": [{"name": "CAM_FRONT", "K": [[3x3]], "T_cam2global": [16],
             "width": 1600, "height": 900}, ...],
  "output_dir": ".../blender",
  "engine": "CYCLES", "cycles_device": "CPU", "cycles_samples": 64,
  "lighting": "fixed" | "random",
  "selfcheck_max_pixel_err": 2.0
}

job.json schema (mode="selfcheck"): synthetic-cube intrinsics/axis verification.
Writes result.json next to the job file (or "result_path" if given).
"""
import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------- argv


def _argv_after_dash():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return []


def _parse_args(argv):
    args = {"job": None}
    i = 0
    while i < len(argv):
        if argv[i] == "--job" and i + 1 < len(argv):
            args["job"] = argv[i + 1]
            i += 2
        else:
            i += 1
    return args


# ---------------------------------------------------------------- small math


def _mat4(flat16):
    return np.array(flat16, dtype=float).reshape(4, 4)


def _project_K(pts_world, T_cam2global, K):
    """numpy reference projection -> (N,2) pixels, plus camera-frame z."""
    T_g2c = np.linalg.inv(T_cam2global)
    pts = np.asarray(pts_world, dtype=float)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pc = (T_g2c @ hom.T).T[:, :3]
    proj = pc @ np.asarray(K, dtype=float).T
    pix = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
    return pix, pc[:, 2]


# ---------------------------------------------------------------- bpy helpers


def _import_bpy():
    import bpy  # noqa: WPS433 (only available inside Blender)
    return bpy


def reset_scene(bpy):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    # deterministic soft lighting: white world + one sun
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 0.6


def add_sun(bpy, rotation_euler=(math.radians(50), 0.0, math.radians(30)), energy=3.0):
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = energy
    light_obj = bpy.data.objects.new("Sun", light_data)
    light_obj.rotation_euler = rotation_euler
    bpy.context.collection.objects.link(light_obj)
    return light_obj


def load_object(bpy, path):
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise ValueError(f"unsupported model format: {ext}")
    new_objs = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new_objs if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh imported from {path}")
    return meshes


def mesh_vertices_world(bpy, objs):
    pts = []
    for o in objs:
        mw = o.matrix_world
        for v in o.data.vertices:
            pts.append((mw @ v.co.to_4d())[:3])
    return np.array(pts, dtype=float) if pts else np.zeros((0, 3))


def surface_signature(objs):
    """TWIN of nusc_inject.mesh_points.surface_signature — kept separate because
    this file runs inside Blender's own Python, which cannot import the package.
    Keep the two in sync; pipeline.py compares the two results as a drift guard.

    Computed from world-space triangle areas and centroids (polygons are fan
    triangulated), so it matches trimesh's sum exactly and is unaffected by the
    importer merging duplicate vertices.
    """
    total = 0.0
    acc = np.zeros(3)
    for o in objs:
        mw = o.matrix_world
        world = np.array([(mw @ v.co.to_4d())[:3] for v in o.data.vertices],
                         dtype=float)
        for p in o.data.polygons:
            idx = list(p.vertices)
            for k in range(1, len(idx) - 1):
                tri = world[[idx[0], idx[k], idx[k + 1]]]
                cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                a = 0.5 * float(np.linalg.norm(cross))
                total += a
                acc += tri.mean(axis=0) * a
    if total <= 0:
        raise RuntimeError("mesh has zero surface area")
    return np.concatenate([[total], acc / total]).tolist()


# ---------------------------------------------------------------- intrinsics

# shift candidate formula (resolved by numeric self-check):
#   shift_x = sx * (cx - W/2) / W
#   shift_y = sy * (cy - H/2) / D        D in {W, H}
SHIFT_CANDIDATES = [(sx, sy, d)
                    for sx in (1.0, -1.0)
                    for sy in (1.0, -1.0)
                    for d in ("W", "H")]


def apply_intrinsics(cam_data, K, width, height, sx, sy, denom):
    fx = float(K[0][0])
    cx, cy = float(K[0][2]), float(K[1][2])
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.sensor_height = 36.0 * height / width
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.lens = fx * cam_data.sensor_width / width
    cam_data.shift_x = sx * (cx - width / 2.0) / width
    d = width if denom == "W" else height
    cam_data.shift_y = sy * (cy - height / 2.0) / d
    cam_data.clip_start = 0.1
    cam_data.clip_end = 2000.0


def project_with_bpy(bpy, cam_obj, pts_world):
    from bpy_extras.object_utils import world_to_camera_view
    scene = bpy.context.scene
    pts = np.asarray(pts_world, dtype=float)
    out = np.zeros((pts.shape[0], 2))
    depth = np.zeros(pts.shape[0])
    for i, p in enumerate(pts):
        co = world_to_camera_view(scene, cam_obj, _vec(p))
        out[i, 0] = co.x * scene.render.resolution_x
        out[i, 1] = (1.0 - co.y) * scene.render.resolution_y  # top-left origin
        depth[i] = co.z
    return out, depth


def _vec(p):
    import mathutils
    return mathutils.Vector((float(p[0]), float(p[1]), float(p[2])))


def resolve_intrinsics(bpy, cam_obj, K, width, height, pts_world, pts_pix_ref):
    """Try shift sign/denominator combos; return (sx, sy, denom, max_err_px)."""
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    best = None
    for sx, sy, denom in SHIFT_CANDIDATES:
        apply_intrinsics(cam_obj.data, K, width, height, sx, sy, denom)
        bpy.context.view_layer.update()
        pix, _ = project_with_bpy(bpy, cam_obj, pts_world)
        err = float(np.max(np.linalg.norm(pix - pts_pix_ref, axis=1))) if len(pix) else 1e9
        if best is None or err < best[3]:
            best = (sx, sy, denom, err)
    sx, sy, denom, err = best
    apply_intrinsics(cam_obj.data, K, width, height, sx, sy, denom)
    bpy.context.view_layer.update()
    return best


def set_camera_pose(cam_obj, T_cam2global):
    """nuScenes/OpenCV cam pose (z fwd, y down) -> Blender cam matrix_world."""
    T = np.asarray(T_cam2global, dtype=float) @ np.diag([1.0, -1.0, -1.0, 1.0])
    import mathutils
    cam_obj.matrix_world = mathutils.Matrix(T.tolist())


# ---------------------------------------------------------------- render


def render_one(bpy, job, cam_spec, obj_meshes, output_dir):
    """Render RGBA foreground for one camera. Returns result dict entry."""
    width, height = int(cam_spec["width"]), int(cam_spec["height"])
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.cycles.samples = int(job.get("cycles_samples", 64))
    scene.render.film_transparent = True

    cam_data = bpy.data.cameras.new("Cam_" + cam_spec["name"])
    cam_obj = bpy.data.objects.new("Cam_" + cam_spec["name"], cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    T_cam2global = _mat4(cam_spec["T_cam2global"])
    K = np.array(cam_spec["K"], dtype=float)
    set_camera_pose(cam_obj, T_cam2global)

    # self-check + intrinsics resolution on the actual object geometry
    pts_world = mesh_vertices_world(bpy, obj_meshes)
    pts_pix_ref, _ = _project_K(pts_world, T_cam2global, K)
    sx, sy, denom, err_px = resolve_intrinsics(
        bpy, cam_obj, K, width, height, pts_world, pts_pix_ref
    )

    # 2D bbox from projected vertices (camera-view bounds)
    inside = pts_pix_ref[(pts_pix_ref[:, 0] >= -width) & (pts_pix_ref[:, 0] <= 2 * width)
                         & (pts_pix_ref[:, 1] >= -height) & (pts_pix_ref[:, 1] <= 2 * height)]
    if len(inside):
        bbox2d = [float(inside[:, 0].min()), float(inside[:, 1].min()),
                  float(inside[:, 0].max()), float(inside[:, 1].max())]
    else:
        bbox2d = None

    out_png = os.path.join(output_dir, f"{cam_spec['name']}.png")
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)

    return {
        "name": cam_spec["name"],
        "png": out_png,
        "bbox2d_px": bbox2d,
        "shift": {"sx": sx, "sy": sy, "denom": denom},
        "selfcheck_max_pixel_err": err_px,
    }


# ---------------------------------------------------------------- selfcheck

def run_selfcheck(bpy, job, result_path):
    """Synthetic cube + known K: verify intrinsics mapping and axis convention."""
    reset_scene(bpy)
    add_sun(bpy)

    width, height = 1600, 900
    K = [[1266.417, 0.0, 816.267], [0.0, 1266.417, 491.507], [0.0, 0.0, 1.0]]
    # camera at global origin looking +X (nuScenes/OpenCV convention: z fwd)
    # OpenCV cam z-forward = global +X -> R maps cam(z) to world(x)
    R = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])  # cam x -> world -y, cam y -> world -z (y-down)
    T_cam2global = np.eye(4)
    T_cam2global[:3, :3] = R
    T_cam2global[:3, 3] = [0.0, 0.0, 1.5]

    # synthetic cube 12m ahead of camera
    import mathutils
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(12.0, 0.0, 1.5))
    cube = bpy.context.active_object
    cube.matrix_world = mathutils.Matrix.Translation((12.0, 0.0, 1.5))

    pts_world = mesh_vertices_world(bpy, [cube])
    pts_pix_ref, depths = _project_K(pts_world, T_cam2global, K)

    cam_data = bpy.data.cameras.new("SelfCam")
    cam_obj = bpy.data.objects.new("SelfCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    set_camera_pose(cam_obj, T_cam2global)

    sx, sy, denom, err_px = resolve_intrinsics(
        bpy, cam_obj, K, width, height, pts_world, pts_pix_ref
    )
    max_err_allowed = float(job.get("selfcheck_max_pixel_err", 2.0))

    # also render the cube and compare silhouette bbox with projected corners
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.cycles.samples = 32
    out_png = os.path.join(os.path.dirname(result_path) or ".", "selfcheck_cube.png")
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)

    silhouette_err = _silhouette_bbox_err(out_png, pts_pix_ref)

    passed = err_px <= max_err_allowed and silhouette_err is not None and silhouette_err <= 3.0
    result = {
        "mode": "selfcheck",
        "passed": bool(passed),
        "shift": {"sx": sx, "sy": sy, "denom": denom},
        "selfcheck_max_pixel_err": err_px,
        "silhouette_bbox_err_px": silhouette_err,
        "selfcheck_png": out_png,
        "max_err_allowed": max_err_allowed,
    }
    _write_json(result_path, result)
    return result


def mathutils_matrix_from_numpy(arr):
    import mathutils
    return mathutils.Matrix(np.asarray(arr, dtype=float).tolist())


def _silhouette_bbox_err(png_path, pts_pix_ref):
    """Compare alpha-mask bbox with projected-vertex bbox. Returns max edge err (px)."""
    try:
        import bpy
        img = bpy.data.images.load(png_path)
        w, h = img.size
        px = np.array(img.pixels[:], dtype=float).reshape(h, w, 4)
        alpha = px[:, :, 3]
        rows, cols = np.where(alpha > 0.05)
        if len(rows) == 0:
            return None
        # bpy pixels row 0 = bottom; convert to top-left origin
        ys = h - 1 - rows
        got = np.array([cols.min(), ys.min(), cols.max(), ys.max()], dtype=float)
        ref = np.array([pts_pix_ref[:, 0].min(), pts_pix_ref[:, 1].min(),
                        pts_pix_ref[:, 0].max(), pts_pix_ref[:, 1].max()], dtype=float)
        return float(np.max(np.abs(got - ref)))
    except Exception:
        return None


# ---------------------------------------------------------------- main


def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    args = _parse_args(_argv_after_dash())
    if not args["job"]:
        raise SystemExit("usage: blender --background --python blender_render.py -- --job job.json")
    with open(args["job"]) as f:
        job = json.load(f)
    result_path = job.get("result_path") or os.path.join(
        job.get("output_dir", os.path.dirname(args["job"])), "result.json"
    )

    bpy = _import_bpy()
    mode = job.get("mode", "render")

    if mode == "selfcheck":
        result = run_selfcheck(bpy, job, result_path)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit("SELF-CHECK FAILED")
        return

    # ---- render mode (supports multi-object job["objects"], legacy single keys) ----
    reset_scene(bpy)
    add_sun(bpy)
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if job.get("objects"):
        object_specs = job["objects"]
    else:
        object_specs = [{"model_path": job["model_path"],
                         "T_model2global": job["T_model2global"]}]

    obj_meshes = []           # all meshes of all objects (flat)
    imported_bboxes = []      # per-object imported-geometry bbox (raw frame)
    for spec in object_specs:
        meshes = load_object(bpy, spec["model_path"])
        pts_imp = mesh_vertices_world(bpy, meshes)
        imported_bboxes.append({
            "model_path": spec["model_path"],
            "min": pts_imp.min(axis=0).tolist(),
            "max": pts_imp.max(axis=0).tolist(),
            "vsig": surface_signature(meshes),
        })
        T_full = mathutils_matrix_from_numpy(_mat4(spec["T_model2global"]))
        for o in meshes:
            o.matrix_world = T_full @ o.matrix_world  # imported geo -> global truth
        obj_meshes.extend(meshes)
    bpy.context.view_layer.update()

    cam_results = []
    for cam_spec in job["cams"]:
        r = render_one(bpy, job, cam_spec, obj_meshes, output_dir)
        cam_results.append(r)
        print("rendered", r["name"], "err_px=%.3f" % r["selfcheck_max_pixel_err"])

    max_err = max((r["selfcheck_max_pixel_err"] for r in cam_results), default=0.0)
    passed = max_err <= float(job.get("selfcheck_max_pixel_err", 2.0))
    result = {
        "mode": "render",
        "passed": bool(passed),
        "cams": cam_results,
        "max_pixel_err": max_err,
        "model_path": job.get("model_path"),
        "imported_bboxes_raw": imported_bboxes,
        # backward-compat single-object key
        "imported_bbox_raw": imported_bboxes[0] if len(imported_bboxes) == 1 else None,
    }
    _write_json(result_path, result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("RENDER SELF-CHECK FAILED (intrinsics mismatch)")


if __name__ == "__main__":
    main()
