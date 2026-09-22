# AdaGScale: Viewpoint-Adaptive Gaussian Scaling in 3D Gaussian Splatting to Reduce Gaussian-Tile Pairs

**Joongho Jo, Hyerin Lim, Hanjun Choi, and Jongsun Park** (Korea University)

[Paper (arXiv)](https://arxiv.org/abs/2604.18980) | [PDF](https://arxiv.org/pdf/2604.18980)

## Abstract

Reducing the number of Gaussian-tile pairs is one of the most promising
approaches to improve 3D Gaussian Splatting (3D-GS) rendering speed on GPUs.
However, the importance difference existing among Gaussian-tile pairs has never
been considered in the previous works. In this paper, we propose AdaGScale, a
novel viewpoint-adaptive Gaussian scaling technique for reducing the number of
Gaussian-tile pairs. AdaGScale is based on the observation that the peripheral
tiles located far from Gaussian center contribute negligibly to pixel color
accumulation. This suggests an opportunity for reducing the number of
Gaussian-tile pairs based on color contribution. AdaGScale efficiently estimates
the color contribution in the peripheral region of each Gaussian during a
preprocessing stage and adaptively scales its size based on the peripheral
score. As a result, Gaussians with lower importance intersect with fewer tiles
during the intersection test, which improves rendering speed while maintaining
image quality. The adjusted size is used only for tile intersection test, and
the original size is retained during color accumulation to preserve visual
fidelity. Experimental results show that AdaGScale achieves a geometric mean
speedup of 13.8× over original 3D-GS on a GPU, with only about 0.5 dB
degradation in PSNR on city-scale scenes.

## Code

This repository provides the AdaGScale source code and precomputed `T_upper`
tables.

New users can follow the complete [first-run guide](docs/QUICKSTART.md), from
Conda environment creation to checking actual and oracle results.

## Quick start

```bash
cd ~/AdaGScale_GitHub
conda env create -f environment.yml
conda activate adagscale
bash build_extension.sh

export MODEL_ROOT=/path/to/pretrained_github
export DATA_ROOT=/path/to/datasets
bash run_official_scene.sh train
```

`MODEL_ROOT` contains a `train/` or `train_ori/` model directory with
`cameras.json` and `point_cloud/iteration_30000/point_cloud.ply`.
`DATA_ROOT` contains `tandt/train/images` and `tandt/train/sparse/0`.
For other scenes, replace `train` with one of the 12 other [supported scene
names](docs/DATA_LAYOUT.md). The script creates the camera lists and config,
checks the paths and image sizes, then runs the actual and oracle protocols
with a default 0.5 dB target. Download links and alternative dataset layouts
are in the [first-run guide](docs/QUICKSTART.md).

AdaGScale uses a scene-specific, 20-bin depth/transmittance upper-bound table
(`T_upper`) and a calibrated peripheral-loss parameter `K` to reduce
Gaussian–tile pairs. This release includes **precomputed tables**, the renderer,
and scripts for K calibration and evaluation. See
[`tupper_tables/README.md`](tupper_tables/README.md) for table provenance and
the exact 16 training views selected for each supplied table.

## Contents

| Path | Purpose |
| --- | --- |
| `csrc/`, `setup.py`, `renderer.py` | FlashGS CUDA renderer with AdaGScale preprocessing |
| `tupper_tables/`, `tupper.py` | Scene-specific precomputed `T_upper` tables and loader |
| `prepare_official_3dgs.py` | Make train/test camera lists from an official 3D-GS checkpoint and COLMAP dataset |
| `run_official_scene.sh` | Prepare, validate, and evaluate one official 3D-GS scene by name |
| `prepare_camera_splits.py` | Make disjoint lists from other existing camera JSON files |
| `calibrate_k.py` | Find K for a target mean PSNR drop on a labeled split |
| `benchmark_scene.py` | Measure FlashGS and AdaGScale FPS, pairs, and test PSNR |
| `run_experiment.py` | Run separate actual/oracle protocols from a scene config |
| `docs/DATA_LAYOUT.md` | Dataset tree, path setup, and split examples |

The `adagscale_cuda` extension contains the FlashGS baseline,
sorting/rasterization, and the `preprocess_adagscale` depth-LUT path. Unused
research kernels and bindings have been removed. The vendored
`csrc/cuda_rasterizer/render.cu` includes a later memory-access correction;
results may differ slightly from the original paper run. Use this same renderer
version for both baseline and AdaGScale comparisons.

## Environment and build details

The paper used an NVIDIA RTX A6000, GCC 11.4, and CUDA 11.6.
`environment.yml` lists the Python, PyTorch, CUDA runtime, CUDA compiler, and
build dependencies for this release. It serves the role of both an environment
file and a Python requirements list. See the [first-run guide](docs/QUICKSTART.md)
for a complete installation and a solution for slow Conda solving.
MKL is constrained below 2024.1 because newer releases prevent this PyTorch
build from importing.

The Conda environment also installs GCC/G++ 10.4 for CUDA compilation; the build
script uses a system `g++` for Python binding and linking. This avoids a CUDA
11.6 compilation failure seen with the system GCC 11 on the author's machine.

Build the extension on the target machine. GPU binaries, model checkpoints,
and dataset images are not included.

## Inputs and paths

Each scene needs a Gaussian `point_cloud.ply`, a matching `T_upper` CSV,
disjoint train/test camera records, and ground-truth images. The official 3D-GS
model directory contains `cameras.json` and
`point_cloud/iteration_30000/point_cloud.ply`; the source dataset contains
`images` (or `images_2`/`images_4`) and `sparse/0`. The official distribution
does **not** require `cameras_Flash.json` or `cameras_3DGS.json`. See the
[first-run guide](docs/QUICKSTART.md) for downloads and complete commands.

After downloading the official 3D-GS models and source datasets, point to the
directories containing the scene folders. Run any of the 13 official scenes by
name:

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
bash run_official_scene.sh train --validate-only
bash run_official_scene.sh train --protocol both --target-drop 0.5
bash run_official_scene.sh truck --protocol both --target-drop 0.5
```

`DATA_ROOT` contains `tandt/`, `db/`, and the nine Mip-NeRF 360 scene folders.
If these groups are elsewhere, set `TANDT_ROOT`, `DB_ROOT`, or
`MIPNERF360_ROOT` separately. The script selects the matching image resolution,
creates disjoint camera lists and a scene config under `local/official_3dgs/`,
and uses the exact 16 training views listed in the supplied T_upper manifest.
It validates inputs before running either protocol. See
[dataset layout](docs/DATA_LAYOUT.md) for all 13 scene names and folder paths.

For a checkpoint outside the official Original 3D-GS set, see
[custom model paths](docs/DATA_LAYOUT.md).
For the supplied CityGaussian large scenes (`building`, `rubble`, `residence`,
and `sciart`), set `"tile_size": 32` in the scene config and use the matching
`*_citygs_large_32x32.csv` table. The renderer then uses `render_32x32`.
The supplied Original 3D-GS scene script uses 16×16 tiles.

## Actual and oracle speed measurements

Both protocols use the **same supplied scene `T_upper` table** and measure FPS
and PSNR on the **test set**. Only the views used to choose K differ:

| Protocol | K calibration views | FPS and PSNR views | Interpretation |
| --- | --- | --- | --- |
| `actual` | 16 training views by default | held-out test views | Performance without test-set tuning |
| `oracle` | all test views | the same test views | Matched-quality speed comparison with test-set K tuning |

The actual result reports both the target calibration drop and the **measured
test PSNR drop**. These may differ. Oracle tuning uses test ground truth to
choose K, so label those numbers as oracle when publishing them. The paper's
matched-quality speed figure uses this oracle setup; its held-out PSNR analysis
uses train-calibrated K.

```bash
bash run_official_scene.sh train --protocol both --target-drop 0.5
```

Separate `results/<case_id>/actual_.../` and `oracle_.../` directories contain
`calibration.csv`, `benchmark.json`, and `summary.json`. The summary records
the K-selection split, calibration-view count, test PSNR drop, FPS, speedup,
and table hash. Default timing uses two full test-set warmup passes followed
by one timed full pass in each of seven trials. Use `--max-pairs-power` if a different pair-buffer
capacity is needed and report that change.

The provided `T_upper` tables use 16 training views, as specified in the paper.
`run_official_scene.sh` generates `calibration_16.json` from the exact names in
the table manifest and uses it for the actual protocol. For a custom config,
set `train.calibration_cameras_json` to a preselected JSON list, or let the
runner choose 16 views at a fixed interval.

For manual runs, use `calibrate_k.py --split-label train` or
`--split-label test --all-views` with the corresponding camera and GT paths,
then pass the returned **K value** to `benchmark_scene.py --k`. A 0.5 dB target
is usually a different number from `K=0.5`. The paper's threshold includes
`K/(2π)`; `renderer.py` applies that conversion before the CUDA call. The
`kernel_input_param` result field gives the value comparable to the original
research scripts' raw CUDA parameter.

The PLY-based benchmark supports Original 3D-GS, GaussianSpa, and CityGaussian
checkpoints with the FlashGS PLY layout. Scaffold-GS generates Gaussians
dynamically and requires a separate pipeline adapter, which is not included.

## Attribution and licenses

FlashGS and vendored GLM notices are in `third_party/`; see `NOTICE.md`.
Model and dataset licenses remain with their providers. A repository-level
license for the original AdaGScale additions can be added if you want to grant
reuse rights for that code.
