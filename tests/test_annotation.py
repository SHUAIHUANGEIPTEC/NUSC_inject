"""Annotation specs and deterministic token generation."""
from __future__ import annotations

import json

import pytest

from nusc_inject import annotation as ann_mod

SAMPLE = {"token": "sample_tok", "scene_token": "scene_tok", "timestamp": 123}


def test_det_token_is_stable_and_token_shaped():
    a = ann_mod.det_token("instance", "k1")
    assert a == ann_mod.det_token("instance", "k1")
    assert len(a) == 32
    assert a != ann_mod.det_token("instance", "k2")
    assert a != ann_mod.det_token("annotation", "k1")   # namespace matters


def test_build_sample_spec_carries_geometry_and_point_counts():
    gts = [{"translation": [1.0, 2.0, 3.0], "size": [1.0, 2.0, 3.0],
            "rotation": [1.0, 0.0, 0.0, 0.0], "category": "anomaly",
            "model_path": "m.glb", "ground_z_lidar": -1.7}]
    lidar_result = {"num_obj_kept_per_object": [42]}
    files = {"lidar_filename": "samples/LIDAR_TOP/x.pcd.bin",
             "lidar_output_path": "/out/x.bin",
             "cam_filenames": {"CAM_FRONT": "samples/CAM_FRONT/x.jpg"},
             "cam_output_paths": {"CAM_FRONT": "/out/x.jpg"}}

    spec = ann_mod.build_sample_spec(SAMPLE, gts, lidar_result, files,
                                     track_keys=["track0"])

    assert spec["sample_token"] == "sample_tok"
    assert spec["scene_token"] == "scene_tok"
    assert spec["timestamp"] == 123
    obj = spec["objects"][0]
    assert obj["num_lidar_pts"] == 42
    assert obj["track_key"] == "track0"
    assert obj["translation"] == [1.0, 2.0, 3.0]


def test_build_sample_spec_without_track_keys_still_works():
    gts = [{"translation": [0.0] * 3, "size": [1.0] * 3,
            "rotation": [1.0, 0.0, 0.0, 0.0], "category": "anomaly",
            "model_path": "m.glb", "ground_z_lidar": 0.0}]
    spec = ann_mod.build_sample_spec(SAMPLE, gts, {"num_obj_kept_per_object": [1]},
                                     {"cam_filenames": {}, "cam_output_paths": {}})
    assert spec["objects"][0]["track_key"] is None
    assert spec["lidar_filename"] is None


def test_ensure_category_is_idempotent():
    categories = []
    tok = ann_mod._ensure_category(categories, "movable_object.anomaly", "d")
    assert len(categories) == 1
    assert ann_mod._ensure_category(categories, "movable_object.anomaly", "d") == tok
    assert len(categories) == 1


def test_ensure_attribute_is_idempotent():
    attributes = []
    tok = ann_mod._ensure_attribute(attributes, "vehicle.moving", "d")
    assert ann_mod._ensure_attribute(attributes, "vehicle.moving", "d") == tok
    assert len(attributes) == 1


def test_ensure_category_reuses_an_existing_row():
    categories = [{"token": "orig", "name": "car", "description": ""}]
    assert ann_mod._ensure_category(categories, "car", "d") == "orig"
    assert len(categories) == 1


def test_patch_devkit_colormap_registers_missing_category():
    class Nusc:
        colormap = {"car": (255, 158, 0)}

    nusc = Nusc()
    ann_mod.patch_devkit_colormap(nusc, "movable_object.anomaly")
    assert nusc.colormap["movable_object.anomaly"] == (255, 0, 255)
    assert nusc.colormap["car"] == (255, 158, 0)


def test_patch_devkit_colormap_does_not_override_existing_entry():
    class Nusc:
        colormap = {"movable_object.anomaly": (1, 2, 3)}

    nusc = Nusc()
    ann_mod.patch_devkit_colormap(nusc, "movable_object.anomaly")
    assert nusc.colormap["movable_object.anomaly"] == (1, 2, 3)


def test_tables_list_covers_the_full_nuscenes_metadata_set():
    assert set(ann_mod.TABLES) == {
        "attribute", "calibrated_sensor", "category", "ego_pose", "instance",
        "log", "map", "sample_annotation", "sample_data", "sample", "scene",
        "sensor", "visibility",
    }


def test_export_refuses_to_overwrite_the_source_dataset(tmp_path):
    import pytest
    cfg = {"dataset": {"root": str(tmp_path), "version": "v1.0-mini"},
           "annotation": {}}
    with pytest.raises(ValueError, match="must differ"):
        ann_mod.export_dataset(cfg, [], tmp_path)


def _fake_dataset(root):
    """Minimal nuScenes dataroot: 13 metadata tables + an empty samples/ tree."""
    meta = root / "v1.0-mini"
    meta.mkdir(parents=True)
    empty = ("attribute", "calibrated_sensor", "category", "ego_pose", "instance",
             "log", "map", "sample_annotation", "sample_data", "sample", "scene",
             "sensor")
    for name in empty:
        (meta / f"{name}.json").write_text("[]")
    (meta / "visibility.json").write_text('[{"token": "4"}]')
    (root / "samples").mkdir()
    return {"dataset": {"root": str(root), "version": "v1.0-mini"},
            "annotation": {"visibility_token": "4"}}


def test_export_rejects_a_filename_outside_samples(tmp_path):
    cfg = _fake_dataset(tmp_path)
    spec = {"sample_token": "s", "scene_token": "sc", "timestamp": 0,
            "lidar_filename": "sweeps/LIDAR_TOP/x.bin",
            "lidar_output_path": None, "cam_filenames": {}, "cam_output_paths": {},
            "objects": []}
    with pytest.raises(ValueError, match="not under samples/"):
        ann_mod.export_dataset(cfg, [spec], tmp_path / "export")


def test_export_rejects_an_unknown_visibility_token(tmp_path):
    cfg = _fake_dataset(tmp_path)
    cfg["annotation"]["visibility_token"] = "9"
    with pytest.raises(ValueError, match="not in visibility.json"):
        ann_mod.export_dataset(cfg, [], tmp_path / "export")


def test_export_adds_one_instance_and_annotation_per_object(tmp_path):
    cfg = _fake_dataset(tmp_path)
    spec = {"sample_token": "s", "scene_token": "sc", "timestamp": 0,
            "lidar_filename": None, "lidar_output_path": None,
            "cam_filenames": {}, "cam_output_paths": {},
            "objects": [{"translation": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
                         "rotation": [1.0, 0.0, 0.0, 0.0], "category": "anomaly",
                         "model_path": "m.glb", "num_lidar_pts": 7,
                         "ground_z_lidar": 0.0, "track_key": None}]}
    out = ann_mod.export_dataset(cfg, [spec], tmp_path / "export")

    assert out["n_new_annotations"] == 1
    anns = json.loads((tmp_path / "export" / "v1.0-mini" /
                       "sample_annotation.json").read_text())
    assert len(anns) == 1
    assert anns[0]["num_lidar_pts"] == 7
    assert anns[0]["prev"] == "" and anns[0]["next"] == ""

    cats = json.loads((tmp_path / "export" / "v1.0-mini" /
                       "category.json").read_text())
    assert [c["name"] for c in cats] == ["movable_object.anomaly"]


def test_export_chains_tracked_objects_across_frames(tmp_path):
    cfg = _fake_dataset(tmp_path)
    specs = []
    for i in range(3):
        specs.append({
            "sample_token": f"s{i}", "scene_token": "sc", "timestamp": i,
            "lidar_filename": None, "lidar_output_path": None,
            "cam_filenames": {}, "cam_output_paths": {},
            "objects": [{"translation": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
                         "rotation": [1.0, 0.0, 0.0, 0.0], "category": "anomaly",
                         "model_path": "m.glb", "num_lidar_pts": 3,
                         "ground_z_lidar": 0.0, "track_key": "track0"}]})
    out = ann_mod.export_dataset(cfg, specs, tmp_path / "export")
    assert out["n_new_annotations"] == 3

    meta = tmp_path / "export" / "v1.0-mini"
    insts = json.loads((meta / "instance.json").read_text())
    anns = json.loads((meta / "sample_annotation.json").read_text())
    assert len(insts) == 1 and insts[0]["nbr_annotations"] == 3
    by_token = {a["token"]: a for a in anns}
    assert by_token[insts[0]["first_annotation_token"]]["prev"] == ""
    assert by_token[insts[0]["last_annotation_token"]]["next"] == ""
    # the middle annotation links to both neighbours
    middle = by_token[insts[0]["first_annotation_token"]]["next"]
    assert by_token[middle]["prev"] == insts[0]["first_annotation_token"]
