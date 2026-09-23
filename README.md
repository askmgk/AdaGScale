# AdaGScale: Viewpoint-Adaptive Gaussian Scaling in 3D Gaussian Splatting to Reduce Gaussian-Tile Pairs

**Joongho Jo, Hyerin Lim, Hanjun Choi, and Jongsun Park** (Korea University)

**Accepted at DAC 2026.**

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

This repository provides an example AdaGScale implementation and the
precomputed `T_upper` tables needed to apply it to pretrained checkpoints
released by the [official 3D Gaussian Splatting project](https://github.com/graphdeco-inria/gaussian-splatting).

A custom checkpoint requires a newly computed `T_upper` table matched to that
checkpoint, its camera setup, image resolution, and renderer tile size. The
`T_upper` extraction code is not included in this release.

## Quick start

```bash
git clone https://github.com/askmgk/AdaGScale.git AdaGScale_GitHub
cd AdaGScale_GitHub
conda env create -f environment.yml
conda activate adagscale
bash build_extension.sh

export MODEL_ROOT=/path/to/pretrained_models
export DATA_ROOT=/path/to/datasets
bash run_official_scene.sh train
```

The command above prepares the official `train` scene and runs both evaluation
protocols with a target PSNR drop of 0.5 dB. Use `--validate-only` to check the
model and dataset paths without rendering:

```bash
bash run_official_scene.sh train --validate-only
```

See the [first-run guide](docs/QUICKSTART.md) for downloads and complete setup
instructions, and [dataset layout](docs/DATA_LAYOUT.md) for supported scenes and
path configuration.

## Evaluation protocols

Both protocols measure FPS and PSNR on the held-out test views. They differ only
in the views used to select `K`:

| Protocol | Views used to select `K` | Interpretation |
| --- | --- | --- |
| `actual` | 16 training views | Test performance without test-set tuning |
| `oracle` | All test views | Matched-quality speed comparison with test-set tuning |

The supplied `T_upper` table is unchanged between the two protocols and is
computed from 16 training views. `--target-drop 0.5` specifies a target PSNR
drop in dB; it does not set `K` to 0.5.

Results are written separately under `results/<case_id>/actual_.../` and
`results/<case_id>/oracle_.../`. Each directory contains `calibration.csv`,
`benchmark.json`, and `summary.json`.

## Supplied tables and supported checkpoints

The repository includes precomputed tables for the 13 Original 3D-GS
checkpoints supported by the one-command `run_official_scene.sh` workflow:

`train`, `truck`, `drjohnson`, `playroom`, `bicycle`, `flowers`, `garden`,
`stump`, `treehill`, `bonsai`, `counter`, `kitchen`, and `room`.

These tables use 16×16 tiles. Other pretrained or custom checkpoints require a
newly computed table matched to the checkpoint and rendering configuration. See
[`tupper_tables/README.md`](tupper_tables/README.md) for compatibility details.

## Repository structure

| Path | Purpose |
| --- | --- |
| `csrc/`, `setup.py`, `renderer.py` | FlashGS CUDA renderer with AdaGScale preprocessing |
| `tupper_tables/`, `tupper.py` | Precomputed `T_upper` tables and loader |
| `run_official_scene.sh` | Prepare, validate, and evaluate one official 3D-GS scene |
| `prepare_official_3dgs.py` | Build train/test camera lists from official model and COLMAP data |
| `calibrate_k.py` | Select `K` for a target mean PSNR drop |
| `benchmark_scene.py` | Measure FPS, pair counts, and test PSNR |
| `run_experiment.py` | Run the `actual` and `oracle` protocols |

## Reproducibility notes

The paper experiments used an NVIDIA RTX A6000 and CUDA 11.6. Model
checkpoints and dataset images are not included in this repository.

The included renderer contains a memory-access correction applied after the
paper experiments, so exact values may differ slightly from the paper. Use this
same renderer for both the FlashGS baseline and AdaGScale when comparing them.

## Attribution

The CUDA renderer is based on
[FlashGS](https://github.com/InternLandMark/FlashGS), and GLM headers are
vendored under `csrc/glm/`. See [NOTICE.md](NOTICE.md) and `third_party/` for
source provenance and third-party licenses. Model and dataset licenses remain
with their providers.
