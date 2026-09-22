#!/usr/bin/env python3
"""Benchmark one Gaussian PLY, camera set, and one or more fixed K values."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch

from renderer import (Camera, Rasterizer, Scene, measure_fps,
                      measure_pairs, measure_quality, system_info)
from tupper import load_tupper_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ply", type=Path, required=True)
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--tupper-csv", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, choices=[16, 32], required=True)
    parser.add_argument("--k", type=float, nargs="+", required=True,
                        help="Fixed calibrated K values to compare with FlashGS")
    parser.add_argument("--gt-dir", type=Path,
                        help="Optional RGB ground truth for mean per-view PSNR")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Full camera-list warmup passes before the first trial")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Full camera-list passes timed per trial")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--max-pairs-power", type=int, default=27)
    parser.add_argument("--max-tiles-power", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-json", type=Path,
                        help="Exact result file path; overrides --output-dir")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.trials < 1:
        parser.error("warmup must be nonnegative; iterations and trials must be positive")
    if any(k < 0 for k in args.k):
        parser.error("K must be nonnegative")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda:%d" % args.device)
    with args.cameras_json.open() as handle:
        cameras = [Camera(record) for record in json.load(handle)]
    if not cameras:
        raise ValueError("Camera JSON is empty")
    scene = Scene(args.model_ply, device)
    rasterizer = Rasterizer(scene, args.tile_size,
                            2 ** args.max_pairs_power, 2 ** args.max_tiles_power,
                            load_tupper_csv(args.tupper_csv))
    variants = [("baseline", None)] + [("adagscale", k) for k in args.k]
    pair_counts = {variant: measure_pairs(rasterizer, scene, cameras, variant[0],
                                          0.0 if variant[1] is None else variant[1])
                   for variant in variants}
    trials = {variant: [] for variant in variants}
    for trial in range(args.trials):
        order = variants if trial % 2 == 0 else list(reversed(variants))
        for mode, k in order:
            fps = measure_fps(rasterizer, scene, cameras, mode,
                              0.0 if k is None else k,
                              args.warmup if trial == 0 else 0, args.iterations)
            trials[(mode, k)].append(fps)
            print("trial=%d mode=%s K=%s fps=%.3f" %
                  (trial + 1, mode, "-" if k is None else k, fps), flush=True)

    quality = {}
    if args.gt_dir:
        quality = measure_quality(rasterizer, scene, cameras,
                                  [(mode, k, 0.0 if k is None else k)
                                   for mode, k in variants], args.gt_dir, device)
    baseline_fps = statistics.mean(trials[("baseline", None)])
    baseline_pairs = pair_counts[("baseline", None)]
    baseline_psnr = quality.get(("baseline", None))
    rows = []
    for mode, k in variants:
        fps = statistics.mean(trials[(mode, k)])
        psnr = quality.get((mode, k))
        rows.append({
            "mode": mode, "k": k, "mean_fps": fps,
            "median_fps": statistics.median(trials[(mode, k)]),
            "speedup_vs_flashgs": fps / baseline_fps,
            "mean_pairs": pair_counts[(mode, k)],
            "pair_reduction_percent": 100 * (1 - pair_counts[(mode, k)] / baseline_pairs),
            "mean_psnr_db": psnr,
            "psnr_drop_db": None if psnr is None else baseline_psnr - psnr,
            "fps_trials": trials[(mode, k)],
        })
    output = (args.output_json if args.output_json is not None else
              args.output_dir / ("scene_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": system_info(device),
        "model_ply": str(args.model_ply), "cameras_json": str(args.cameras_json),
        "gt_dir": None if args.gt_dir is None else str(args.gt_dir),
        "camera_count": len(cameras), "tile_size": args.tile_size,
        "warmup": args.warmup, "iterations": args.iterations, "trials": args.trials,
        "results": rows,
    }
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print("Results: %s" % output)


if __name__ == "__main__":
    main()
