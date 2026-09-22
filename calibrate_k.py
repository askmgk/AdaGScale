#!/usr/bin/env python3
"""Calibrate AdaGScale K on camera views for a target mean PSNR drop."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from renderer import Camera, Rasterizer, Scene, measure_quality
from tupper import load_tupper_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ply", type=Path, required=True)
    parser.add_argument("--cameras-json", type=Path, required=True,
                        help="Camera records for the selected train or test split")
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="RGB images matching the selected camera records")
    parser.add_argument("--tupper-csv", type=Path, required=True,
                        help="Precomputed 20-bin scene LUT")
    parser.add_argument("--split-label", choices=["train", "test"], required=True,
                        help="Label the views used to select K")
    parser.add_argument("--tile-size", type=int, choices=[16, 32], required=True)
    parser.add_argument("--targets", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--all-views", action="store_true",
                        help="Use every camera record (required for test-set oracle)")
    parser.add_argument("--k-max", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--max-pairs-power", type=int, default=27)
    parser.add_argument("--max-tiles-power", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("calibration_results.csv"))
    args = parser.parse_args()
    if args.sample_count < 1 or args.k_max <= 0 or args.tolerance <= 0:
        parser.error("sample-count, k-max, and tolerance must be positive")
    if args.max_iterations < 1 or any(target <= 0 for target in args.targets):
        parser.error("max-iterations and all targets must be positive")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda:%d" % args.device)
    with args.cameras_json.open() as handle:
        records = json.load(handle)
    if not records:
        raise ValueError("Camera JSON is empty")
    if not args.all_views and len(records) < args.sample_count:
        raise ValueError("Fewer cameras than --sample-count")
    # Pass a preselected JSON file to fix the exact calibration views.
    # Start at the first training camera and advance by a fixed interval.
    # With 16 views this matches the bundled Original 3D-GS T_upper manifest.
    indices = (list(range(len(records))) if args.all_views else
               [i * (len(records) // args.sample_count)
                for i in range(args.sample_count)])
    cameras = [Camera(records[i]) for i in indices]
    scene = Scene(args.model_ply, device)
    rasterizer = Rasterizer(scene, args.tile_size,
                            2 ** args.max_pairs_power, 2 ** args.max_tiles_power,
                            load_tupper_csv(args.tupper_csv))

    def psnr(k, mode="adagscale"):
        return measure_quality(rasterizer, scene, cameras,
                               [(mode, None, k)], args.gt_dir, device)[(mode, None)]

    baseline = psnr(0.0, "baseline")
    print("Baseline mean PSNR: %.6f dB (%d %s views)" %
          (baseline, len(cameras), args.split_label), flush=True)
    upper_drop = baseline - psnr(args.k_max)
    if upper_drop < max(args.targets):
        raise RuntimeError("K upper bound %.6g reaches only %.6f dB drop; increase --k-max" %
                           (args.k_max, upper_drop))

    rows = []
    for target in args.targets:
        left, right = 0.0, args.k_max
        best_k, best_drop = 0.0, 0.0
        for iteration in range(1, args.max_iterations + 1):
            k = (left + right) / 2.0
            drop = baseline - psnr(k)
            if abs(drop - target) < abs(best_drop - target):
                best_k, best_drop = k, drop
            print("target=%.3f iteration=%d K=%.8f drop=%.6f dB" %
                  (target, iteration, k, drop), flush=True)
            if abs(drop - target) <= args.tolerance:
                break
            if drop < target:
                left = k
            else:
                right = k
            if right - left < args.tolerance * 0.1:
                break
        rows.append({"calibration_split": args.split_label,
                     "target_drop_db": target, "k": best_k,
                     "kernel_input_param": best_k / (2.0 * math.pi),
                     "calibration_drop_db": best_drop,
                     "baseline_calibration_psnr_db": baseline,
                     "sample_count": len(cameras), "tile_size": args.tile_size})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Calibration: %s" % args.output_csv)


if __name__ == "__main__":
    main()
