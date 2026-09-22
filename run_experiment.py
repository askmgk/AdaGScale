#!/usr/bin/env python3
"""Run distinct train-calibrated (actual) and test-calibrated (oracle) evaluations."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from tupper import load_tupper_csv


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_path(base, value):
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError("Unresolved environment variable in path: %s" % value)
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve()


def camera_records(path):
    with path.open() as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Camera JSON must be a nonempty list: %s" % path)
    names = [Path(record["img_name"]).stem for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image names in camera JSON: %s" % path)
    return records, set(names)


def image_index(directory):
    index = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in index:
                raise ValueError("Duplicate image stem in %s: %s" % (directory, path.stem))
            index[path.stem] = path
    return index


def check_images(label, records, directory):
    index = image_index(directory)
    missing = [record["img_name"] for record in records
               if Path(record["img_name"]).stem not in index]
    if missing:
        raise ValueError("%s ground truth missing %d images in %s; first: %s" %
                         (label, len(missing), directory, missing[0]))
    for record in records:
        path = index[Path(record["img_name"]).stem]
        with Image.open(path) as image:
            expected = (int(record["width"]), int(record["height"]))
            if image.size != expected:
                raise ValueError("%s camera/image resolution mismatch: %s camera=%s image=%s" %
                                 (label, path, expected, image.size))


def load_case(path):
    with path.open() as handle:
        raw = json.load(handle)
    base = path.parent.resolve()
    case = {"case_id": raw["case_id"], "tile_size": int(raw["tile_size"]),
            "model_ply": resolve_path(base, raw["model_ply"]),
            "tupper_csv": resolve_path(base, raw["tupper_csv"])}
    if case["tile_size"] not in (16, 32):
        raise ValueError("tile_size must be 16 or 32")
    for split in ("train", "test"):
        block = raw[split]
        case[split] = {
            "cameras_json": resolve_path(base, block["cameras_json"]),
            "gt_dir": resolve_path(base, block["gt_dir"]),
        }
    if raw["train"].get("calibration_cameras_json"):
        case["train"]["calibration_cameras_json"] = resolve_path(
            base, raw["train"]["calibration_cameras_json"])
    return case


def validate_case(case):
    for path in [case["model_ply"], case["tupper_csv"],
                 case["train"]["cameras_json"], case["test"]["cameras_json"]]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for split in ("train", "test"):
        if not case[split]["gt_dir"].is_dir():
            raise NotADirectoryError(case[split]["gt_dir"])
    load_tupper_csv(case["tupper_csv"])
    train, train_names = camera_records(case["train"]["cameras_json"])
    test, test_names = camera_records(case["test"]["cameras_json"])
    overlap = train_names & test_names
    if overlap:
        raise ValueError("Train and test cameras overlap (%d images, e.g. %s). "
                         "Run prepare_camera_splits.py first." %
                         (len(overlap), sorted(overlap)[0]))
    check_images("train", train, case["train"]["gt_dir"])
    check_images("test", test, case["test"]["gt_dir"])
    calibration = case["train"].get("calibration_cameras_json")
    if calibration:
        if not calibration.is_file():
            raise FileNotFoundError(calibration)
        selected, selected_names = camera_records(calibration)
        if not selected_names <= train_names:
            raise ValueError("Explicit calibration cameras must be a subset of train cameras")
        check_images("train calibration", selected, case["train"]["gt_dir"])
    return len(train), len(test)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="One case with explicit model, LUT, train, and test paths")
    parser.add_argument("--protocol", choices=["actual", "oracle", "both"], default="both")
    parser.add_argument("--target-drop", type=float, default=0.5)
    parser.add_argument("--train-sample-count", type=int, default=16)
    parser.add_argument("--k-max", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--max-pairs-power", type=int, default=27)
    parser.add_argument("--max-tiles-power", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--validate-only", action="store_true",
                        help="Check splits, paths, images, resolution, and LUT without a GPU")
    args = parser.parse_args()
    if args.target_drop <= 0 or args.train_sample_count < 1:
        parser.error("target-drop and train-sample-count must be positive")
    case = load_case(args.config)
    train_count, test_count = validate_case(case)
    print("Validated %s: %d train, %d test views; LUT=%s" %
          (case["case_id"], train_count, test_count, case["tupper_csv"]))
    if args.validate_only:
        return

    protocols = ["actual", "oracle"] if args.protocol == "both" else [args.protocol]
    for protocol in protocols:
        calibration_split = "train" if protocol == "actual" else "test"
        run_dir = (args.output_root / case["case_id"] /
                   ("%s_target%.3f_%s" %
                    (protocol, args.target_drop, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))))
        run_dir.mkdir(parents=True, exist_ok=False)
        calibration_csv = run_dir / "calibration.csv"
        benchmark_json = run_dir / "benchmark.json"
        calibration_cameras = (case["train"].get("calibration_cameras_json",
                             case["train"]["cameras_json"]) if protocol == "actual" else
                             case["test"]["cameras_json"])
        common = ["--model-ply", str(case["model_ply"]),
                  "--tupper-csv", str(case["tupper_csv"]),
                  "--tile-size", str(case["tile_size"]),
                  "--max-pairs-power", str(args.max_pairs_power),
                  "--max-tiles-power", str(args.max_tiles_power),
                  "--device", str(args.device)]
        calibrate = [sys.executable, str(ROOT / "calibrate_k.py"),
                     *common,
                     "--cameras-json", str(calibration_cameras),
                     "--gt-dir", str(case[calibration_split]["gt_dir"]),
                     "--split-label", calibration_split,
                     "--targets", str(args.target_drop),
                     "--k-max", str(args.k_max),
                     "--tolerance", str(args.tolerance),
                     "--max-iterations", str(args.max_iterations),
                     "--output-csv", str(calibration_csv)]
        if protocol == "oracle" or "calibration_cameras_json" in case["train"]:
            calibrate.append("--all-views")
        else:
            calibrate += ["--sample-count", str(args.train_sample_count)]
        subprocess.run(calibrate, check=True)
        with calibration_csv.open(newline="") as handle:
            calibration = next(csv.DictReader(handle))
        k = float(calibration["k"])
        benchmark = [sys.executable, str(ROOT / "benchmark_scene.py"),
                     *common,
                     "--cameras-json", str(case["test"]["cameras_json"]),
                     "--gt-dir", str(case["test"]["gt_dir"]),
                     "--k", str(k), "--warmup", str(args.warmup),
                     "--iterations", str(args.iterations), "--trials", str(args.trials),
                     "--output-json", str(benchmark_json)]
        subprocess.run(benchmark, check=True)
        with benchmark_json.open() as handle:
            measured = json.load(handle)
        baseline = next(row for row in measured["results"] if row["mode"] == "baseline")
        method = next(row for row in measured["results"] if row["mode"] == "adagscale")
        summary = {
            "case_id": case["case_id"], "protocol": protocol,
            "k_selection_split": calibration_split, "timing_and_psnr_split": "test",
            "target_drop_db": args.target_drop, "k": k,
            "kernel_input_param": float(calibration["kernel_input_param"]),
            "calibration_drop_db": float(calibration["calibration_drop_db"]),
            "measured_test_drop_db": method["psnr_drop_db"],
            "test_drop_error_db": method["psnr_drop_db"] - args.target_drop,
            "baseline_test_fps": baseline["mean_fps"],
            "adagscale_test_fps": method["mean_fps"],
            "speedup_vs_flashgs": method["speedup_vs_flashgs"],
            "pair_reduction_percent": method["pair_reduction_percent"],
            "train_view_count": train_count, "test_view_count": test_count,
            "k_calibration_view_count": int(calibration["sample_count"]),
            "tupper_csv": str(case["tupper_csv"]),
            "tupper_sha256": digest(case["tupper_csv"]),
            "calibration_cameras_json": str(calibration_cameras),
            "test_cameras_json": str(case["test"]["cameras_json"]),
            "benchmark_json": str(benchmark_json),
        }
        with (run_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        print("%s result: %s" % (protocol, run_dir / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
