#!/usr/bin/env python3
"""Prepare train/test camera JSONs from an official 3D-GS model and dataset.

The model's cameras.json may contain all views or only test views. Missing
training cameras are reconstructed from the dataset's COLMAP sparse/0 model.
"""

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4)}
ROOT = Path(__file__).resolve().parent


def stem(name):
    return Path(name).stem


def read_exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise ValueError("Truncated COLMAP binary file")
    return data


def read_c_string(handle):
    data = bytearray()
    while True:
        char = read_exact(handle, 1)
        if char == b"\x00":
            return data.decode("utf-8")
        data.extend(char)


def read_cameras_bin(path):
    cameras = {}
    with path.open("rb") as handle:
        count, = struct.unpack("<Q", read_exact(handle, 8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", read_exact(handle, 24))
            if model_id not in CAMERA_MODELS:
                raise ValueError("COLMAP camera model %d is not supported" % model_id)
            model, param_count = CAMERA_MODELS[model_id]
            params = struct.unpack("<" + "d" * param_count,
                                   read_exact(handle, 8 * param_count))
            cameras[camera_id] = (model, width, height, params)
    return cameras


def read_images_bin(path):
    images = {}
    with path.open("rb") as handle:
        count, = struct.unpack("<Q", read_exact(handle, 8))
        for _ in range(count):
            values = struct.unpack("<idddddddi", read_exact(handle, 64))
            name = read_c_string(handle)
            point_count, = struct.unpack("<Q", read_exact(handle, 8))
            handle.seek(point_count * 24, 1)
            key = stem(name)
            if key in images:
                raise ValueError("Duplicate COLMAP image name: %s" % name)
            images[key] = (name, values[1:5], values[5:8], values[8])
    return images


def read_cameras_txt(path):
    cameras = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        camera_id = int(fields[0])
        model = fields[1]
        if model not in ("SIMPLE_PINHOLE", "PINHOLE"):
            raise ValueError("COLMAP camera model %s is not supported" % model)
        cameras[camera_id] = (model, int(fields[2]), int(fields[3]),
                              tuple(float(value) for value in fields[4:]))
    return cameras


def read_images_txt(path):
    images = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 10:
                raise ValueError("Invalid COLMAP images.txt record: %s" % line)
            name = fields[9]
            key = stem(name)
            if key in images:
                raise ValueError("Duplicate COLMAP image name: %s" % name)
            images[key] = (name, tuple(float(x) for x in fields[1:5]),
                           tuple(float(x) for x in fields[5:8]), int(fields[8]))
            next(handle, None)  # The following line contains 2D points.
    return images


def read_colmap(directory):
    cameras_bin = directory / "cameras.bin"
    images_bin = directory / "images.bin"
    if cameras_bin.is_file() and images_bin.is_file():
        return read_cameras_bin(cameras_bin), read_images_bin(images_bin)
    cameras_txt = directory / "cameras.txt"
    images_txt = directory / "images.txt"
    if cameras_txt.is_file() and images_txt.is_file():
        return read_cameras_txt(cameras_txt), read_images_txt(images_txt)
    return None, None


def qvec_to_rotation(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z,
         2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z,
         2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x,
         1 - 2 * x * x - 2 * y * y],
    ])


def colmap_camera_record(source, intrinsics, image_path):
    name, qvec, tvec, camera_id = source
    model, original_width, original_height, params = intrinsics[camera_id]
    if model == "SIMPLE_PINHOLE":
        original_fx = original_fy = params[0]
    else:
        original_fx, original_fy = params[:2]
    with Image.open(image_path) as image:
        width, height = image.size
    world_to_camera = qvec_to_rotation(qvec)
    camera_to_world = world_to_camera.T
    position = -camera_to_world @ np.asarray(tvec)
    return {
        "img_name": name,
        "width": width,
        "height": height,
        "position": position.tolist(),
        "rotation": camera_to_world.tolist(),
        "fx": original_fx * width / original_width,
        "fy": original_fy * height / original_height,
    }


def image_paths(directory):
    paths = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in paths:
                raise ValueError("Duplicate image stem in %s: %s" % (directory, path.stem))
            paths[path.stem] = path
    if not paths:
        raise ValueError("No RGB images found in %s" % directory)
    return paths


def model_camera_records(path):
    with path.open() as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Empty or invalid model cameras.json: %s" % path)
    by_name = {}
    required = {"img_name", "width", "height", "position", "rotation", "fx", "fy"}
    for record in records:
        if not required <= set(record):
            raise ValueError("Model camera record lacks required fields: %s" % record)
        key = stem(record["img_name"])
        if key in by_name:
            raise ValueError("Duplicate model camera name: %s" % key)
        by_name[key] = record
    return by_name


def check_matching_camera(saved, generated, name):
    for field in ("position", "rotation"):
        difference = np.max(np.abs(np.asarray(saved[field]) - np.asarray(generated[field])))
        if difference > 1e-4:
            raise ValueError("Model and dataset %s disagree for %s: %s" %
                             (field, name, difference))
    for field in ("fx", "fy"):
        if abs(saved[field] - generated[field]) > max(1e-3, generated[field] * 1e-4):
            raise ValueError("Model and dataset %s disagree for %s" % (field, name))


def write_scene_config(scene, model_dir, images_dir, output_dir, train):
    case_id = scene + "_ori_small_16x16"
    lut = ROOT / "tupper_tables" / (case_id + ".csv")
    if not lut.is_file():
        raise FileNotFoundError("No supplied T_upper table for %s: %s" % (scene, lut))
    with (ROOT / "tupper_tables" / "MANIFEST.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["case_id"] == case_id]
    if len(rows) != 1 or int(rows[0]["calibration_views"]) != 16:
        raise ValueError("Missing or invalid 16-view T_upper manifest for %s" % scene)
    names = [stem(name) for name in rows[0]["img_names"].split(";")]
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError("Invalid T_upper camera names for %s" % scene)
    by_name = {stem(record["img_name"]): record for record in train}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError("%s: T_upper cameras are absent from the official "
                         "training split: %s" % (scene, missing))
    output_dir = output_dir.resolve()
    model_dir = model_dir.resolve()
    images_dir = images_dir.resolve()
    calibration_path = output_dir / "calibration_16.json"
    with calibration_path.open("w") as handle:
        json.dump([by_name[name] for name in names], handle, indent=2)
    config = {
        "case_id": case_id,
        "model_ply": str(model_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"),
        "tupper_csv": str(lut),
        "tile_size": 16,
        "train": {"cameras_json": str(output_dir / "train.json"),
                  "calibration_cameras_json": str(calibration_path),
                  "gt_dir": str(images_dir)},
        "test": {"cameras_json": str(output_dir / "test.json"),
                 "gt_dir": str(images_dir)},
    }
    config_path = output_dir / "config.json"
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2)
    print("Wrote %s with the manifest's 16 training views" % config_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Official pretrained scene directory with cameras.json")
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="RGB directory in the matching source dataset")
    parser.add_argument("--colmap-dir", type=Path,
                        help="COLMAP sparse/0 directory; defaults to <images-dir>/../sparse/0")
    parser.add_argument("--holdout", type=int, default=8,
                        help="Every Nth sorted registered image is test (official default: 8)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", type=str,
                        help="Official scene name; also write config.json and calibration_16.json")
    args = parser.parse_args()
    if args.holdout < 2:
        parser.error("--holdout must be at least 2")
    model = model_camera_records(args.model_dir / "cameras.json")
    paths = image_paths(args.images_dir)
    colmap_dir = args.colmap_dir or args.images_dir.parent / "sparse" / "0"
    intrinsics, registered = read_colmap(colmap_dir)

    if registered is None:
        if set(model) != set(paths):
            raise ValueError("The model cameras.json does not cover all RGB views. "
                             "Provide the matching COLMAP sparse/0 with --colmap-dir.")
        names = sorted(model, key=lambda key: model[key]["img_name"])
    else:
        names = sorted(registered, key=lambda key: registered[key][0])
        if not set(model) <= set(registered):
            raise ValueError("Model cameras.json contains names outside COLMAP sparse/0")
        missing_images = set(registered) - set(paths)
        if missing_images:
            raise ValueError("RGB image missing for COLMAP camera: %s" %
                             sorted(missing_images)[0])
    if len(names) < args.holdout:
        raise ValueError("Too few registered images for holdout=%d" % args.holdout)

    test_names = set(names[::args.holdout])
    if set(model) != set(names) and set(model) != test_names:
        raise ValueError("Model cameras.json is neither a complete camera list "
                         "nor the official holdout test list. Check the checkpoint "
                         "and --holdout value.")

    train = []
    test = []
    generated_count = 0
    for index, name in enumerate(names):
        with Image.open(paths[name]) as image:
            expected_size = image.size
        if name in model:
            record = dict(model[name])
            if (int(record["width"]), int(record["height"])) != expected_size:
                raise ValueError("Model camera/image resolution mismatch for %s: "
                                 "camera=%s image=%s. Choose the matching images directory." %
                                 (name, (record["width"], record["height"]), expected_size))
            if registered is not None:
                generated = colmap_camera_record(registered[name], intrinsics, paths[name])
                check_matching_camera(record, generated, name)
        else:
            record = colmap_camera_record(registered[name], intrinsics, paths[name])
            generated_count += 1
        record["id"] = index
        (test if name in test_names else train).append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in (("train", train), ("test", test)):
        with (args.output_dir / (split + ".json")).open("w") as handle:
            json.dump(records, handle, indent=2)
    print("Wrote %d train and %d test cameras to %s (%d reconstructed from COLMAP)" %
          (len(train), len(test), args.output_dir, generated_count))
    if args.scene:
        write_scene_config(args.scene, args.model_dir, args.images_dir, args.output_dir, train)


if __name__ == "__main__":
    main()
