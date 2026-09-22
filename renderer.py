"""FlashGS renderer and image-quality helpers for AdaGScale."""

import os
import platform
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
try:
    import adagscale_cuda as cuda_ops
except ImportError:
    cuda_ops = None

def arithmetic_mean(values):
    return sum(values) / len(values)


class Scene:
    def __init__(self, ply_path, device):
        if cuda_ops is None:
            raise RuntimeError("Build the CUDA extension with bash build_extension.sh first")
        values = cuda_ops.ops.loadPly(str(ply_path))
        self.num_vertex = values[0]
        self.position = values[1].to(device)
        self.shs = values[2].to(device)
        self.opacity = values[3].to(device)
        self.cov3d = values[4].to(device)
        self.device = device


class Camera:
    def __init__(self, record):
        self.img_name = record["img_name"]
        self.width = int(record["width"])
        self.height = int(record["height"])
        self.position = torch.tensor(record["position"], dtype=torch.float32)
        self.rotation = torch.tensor(record["rotation"], dtype=torch.float32)
        self.focal_x = float(record["fx"])
        self.focal_y = float(record["fy"])
        self.z_far = 100.0
        self.z_near = 0.01


class Rasterizer:
    def __init__(self, scene, tile_size, max_pairs, max_tiles, tupper_lut):
        device = scene.device
        if len(tupper_lut) != 20:
            raise ValueError("Tupper LUT must have 20 depth-bin values")
        self.tupper_lut = torch.tensor(tupper_lut, device=device, dtype=torch.float32)
        self.tile_size = tile_size
        self.max_pairs = max_pairs
        self.keys_unsorted = torch.zeros(max_pairs, device=device, dtype=torch.int64)
        self.values_unsorted = torch.zeros(max_pairs, device=device, dtype=torch.int32)
        self.keys_sorted = torch.zeros(max_pairs, device=device, dtype=torch.int64)
        self.values_sorted = torch.zeros(max_pairs, device=device, dtype=torch.int32)
        sort_bytes = cuda_ops.ops.get_sort_buffer_size(max_pairs)
        self.sort_space = torch.zeros(sort_bytes, device=device, dtype=torch.int8)
        self.ranges = torch.zeros((max_tiles, 2), device=device, dtype=torch.int32)
        self.curr_offset = torch.zeros(1, device=device, dtype=torch.int32)
        self.points_xy = torch.zeros((scene.num_vertex, 2), device=device, dtype=torch.float32)
        self.rgb_depth = torch.zeros((scene.num_vertex, 4), device=device, dtype=torch.float32)
        self.conic_opacity = torch.zeros((scene.num_vertex, 4), device=device, dtype=torch.float32)

    def forward(self, scene, camera, mode, k, return_output=False):
        self.curr_offset.fill_(0)
        common = (
            scene.position, scene.shs, scene.opacity, scene.cov3d,
            camera.width, camera.height, self.tile_size, self.tile_size,
            camera.position, camera.rotation,
            camera.focal_x, camera.focal_y, camera.z_far, camera.z_near,
            self.points_xy, self.rgb_depth, self.conic_opacity,
            self.keys_unsorted, self.values_unsorted, self.curr_offset,
        )
        if mode == "baseline":
            cuda_ops.ops.preprocess(*common)
        elif mode == "adagscale":
            # The kernel accepts K/(2*pi); the paper defines K before this factor.
            cuda_ops.ops.preprocess_adagscale(
                float(k) / (2.0 * math.pi), self.tupper_lut, *common)
        else:
            raise ValueError("Unknown mode: %s" % mode)

        num_pairs = int(self.curr_offset.cpu()[0])
        if num_pairs >= self.max_pairs:
            raise RuntimeError(
                "Pair capacity exceeded (%d >= %d). Increase --max-pairs-power."
                % (num_pairs, self.max_pairs)
            )

        if num_pairs:
            cuda_ops.ops.sort_gaussian(
                num_pairs, camera.width, camera.height,
                self.tile_size, self.tile_size, self.sort_space,
                self.keys_unsorted, self.values_unsorted,
                self.keys_sorted, self.values_sorted,
            )
        output = torch.zeros(
            (camera.height, camera.width, 3),
            device=scene.device,
            dtype=torch.int8,
        )
        render = cuda_ops.ops.render_16x16 if self.tile_size == 16 else cuda_ops.ops.render_32x32
        background = torch.zeros(3, dtype=torch.float32)
        render(
            num_pairs, camera.width, camera.height,
            self.points_xy, self.rgb_depth, self.conic_opacity,
            self.keys_sorted, self.values_sorted,
            self.ranges, background, output,
        )
        if return_output:
            return num_pairs, output
        return num_pairs


def measure_pairs(rasterizer, scene, cameras, mode, k):
    counts = []
    for camera in cameras:
        counts.append(rasterizer.forward(scene, camera, mode, k))
    torch.cuda.synchronize()
    return arithmetic_mean(counts)


def measure_fps(rasterizer, scene, cameras, mode, k, warmup, iterations):
    for _ in range(warmup):
        for camera in cameras:
            rasterizer.forward(scene, camera, mode, k)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        for camera in cameras:
            rasterizer.forward(scene, camera, mode, k)
    torch.cuda.synchronize()
    return len(cameras) * iterations / (time.perf_counter() - start)


def image_map(directory):
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    return {
        path.stem: path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }


def psnr_for_output(output, gt_path, device):
    gt_array = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
    if tuple(gt_array.shape[:2]) != tuple(output.shape[:2]):
        raise RuntimeError(
            "GT/render size mismatch for %s: GT=%s render=%s"
            % (gt_path, gt_array.shape[:2], tuple(output.shape[:2]))
        )
    gt = torch.from_numpy(gt_array).to(device)
    rendered = output.to(torch.uint8).to(torch.float32) / 255.0
    mse = torch.mean((rendered - gt) ** 2)
    return float((-10.0 * torch.log10(mse)).item())


def measure_quality(rasterizer, scene, cameras, variants, gt_dir, device):
    gt_by_stem = image_map(gt_dir)
    results = {}
    for mode, target, k in variants:
        per_view = []
        for camera in cameras:
            stem = Path(camera.img_name).stem
            if stem not in gt_by_stem:
                raise RuntimeError("Missing GT image for %s in %s" % (camera.img_name, gt_dir))
            _, output = rasterizer.forward(scene, camera, mode, k, return_output=True)
            per_view.append(psnr_for_output(output, gt_by_stem[stem], device))
        results[(mode, target)] = arithmetic_mean(per_view)
    return results


def system_info(device):
    props = torch.cuda.get_device_properties(device)
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "gpu_name": props.name,
        "compute_capability": "%d.%d" % (props.major, props.minor),
        "gpu_memory_gib": props.total_memory / (1024 ** 3),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
    }
