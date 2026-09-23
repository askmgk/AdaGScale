# First run: installation to results

This guide runs the official Original 3D-GS `train` checkpoint. It assumes
Linux, an NVIDIA GPU and driver, Conda, Git, and a system `g++` compiler.

## 1. Clone and create the environment

```bash
git clone https://github.com/askmgk/AdaGScale.git AdaGScale_GitHub
cd AdaGScale_GitHub
nvidia-smi
conda env create -f environment.yml
conda activate adagscale
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The environment uses Python 3.7, PyTorch 1.12.1, and CUDA 11.6. Conda may take
several minutes to resolve and download the environment on a new machine.

## 2. Build the CUDA extension

```bash
bash build_extension.sh
```

A successful build ends with `AdaGScale extension import: OK`. If `nvcc` is
missing, confirm that the `adagscale` environment is active. If CUDA is not
available, check the NVIDIA driver and GPU visibility.

## 3. Download the checkpoint and dataset

This repository does not include pretrained models or dataset images. Download
the following files for the official evaluation scenes:

- [Original 3D-GS pretrained models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)
- [Tanks and Temples / Deep Blending COLMAP data](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)
- [Mip-NeRF 360 datasets](https://jonbarron.info/mipnerf360/)

These resources are also linked by the
[official 3D-GS repository](https://github.com/graphdeco-inria/gaussian-splatting#readme).

For the `train` example, the extracted files must include:

```text
<model root>/train/cameras.json
<model root>/train/point_cloud/iteration_30000/point_cloud.ply
<data root>/tandt/train/images/<RGB images>
<data root>/tandt/train/sparse/0/cameras.bin
<data root>/tandt/train/sparse/0/images.bin
```

The COLMAP files may be `.txt` instead of `.bin`. If the checkpoint's
`cameras.json` contains only held-out cameras, the preparation script
reconstructs the missing training cameras from `sparse/0`.

Set the parent directories containing the model and dataset scene folders:

```bash
export MODEL_ROOT="$HOME/data/models"
export DATA_ROOT="$HOME/data"
```

The script accepts either `$MODEL_ROOT/train` or `$MODEL_ROOT/train_ori` for
this example. See [DATA_LAYOUT.md](DATA_LAYOUT.md) for the complete directory
tree and alternative root variables.

> A `T_upper` table is specific to a checkpoint, camera setup, image
> resolution, and renderer tile size. The included table is intended for the
> corresponding official checkpoint. A custom checkpoint requires a newly
> computed table; extraction code is not included.

## 4. Validate one scene

```bash
bash run_official_scene.sh train --validate-only
```

For the official `train` data, validation reports 263 training and 38 test
views. The command checks the input paths, image sizes, camera split, and
`T_upper` table without running the renderer.

The supported scene names are:

`train`, `truck`, `drjohnson`, `playroom`, `bicycle`, `flowers`, `garden`,
`stump`, `treehill`, `bonsai`, `counter`, `kitchen`, and `room`.

## 5. Run the experiment

```bash
bash run_official_scene.sh train --protocol both --target-drop 0.5
```

Replace `train` with another supported scene name as needed. `actual` selects
`K` from 16 training views; `oracle` selects `K` from the test views. Both then
measure FPS and PSNR on the test views. The value `0.5` is a target PSNR drop
in dB, not the value of `K`.

## 6. Read the results

The two protocols write separate directories under
`results/train_ori_small_16x16/`:

```text
actual_target0.500_.../
oracle_target0.500_.../
```

Each directory contains:

- `summary.json`: selected `K`, measured test PSNR drop, FPS, and speedup;
- `calibration.csv`: the K search history;
- `benchmark.json`: per-trial FPS, pair counts, and PSNR.

Generated camera lists and configuration files are stored under
`local/official_3dgs/<scene>/`. The `local/` and `results/` directories are
ignored by Git.
