# First run: Conda to AdaGScale results

This guide uses the `train_ori_small_16x16` example. Commands assume Linux, a
working NVIDIA GPU driver, Conda, Git, and a system `g++` compiler. The original
paper environment used an RTX A6000 and CUDA 11.6. Run all commands from a
terminal on the machine with the GPU.

## 1. Get the source and make the environment

Clone this repository, then enter it:

```bash
git clone https://github.com/askmgk/AdaGScale.git AdaGScale_GitHub
cd AdaGScale_GitHub
nvidia-smi
conda env create -f environment.yml
conda activate adagscale
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`environment.yml` is the standard, editable specification. It installs
Python 3.7, PyTorch 1.12.1, CUDA 11.6, the CUDA toolkit, compilers, and Python
dependencies. Conda activation must succeed
before building or running the CUDA code. If `conda activate` is unavailable,
initialize Conda for your shell and open a new terminal.
The PyTorch import check catches an incompatible MKL package before the CUDA
build. The YAML constrains MKL below 2024.1 for this reason.

On the author's server, the required packages were already in Conda's cache,
so `conda env create --offline -f environment.yml` also completed quickly.
The offline command requires cached packages; use the normal command on a
fresh machine.

## 2. Build and check the CUDA extension

```bash
bash build_extension.sh
```

Expect `CUDA available: True` and `AdaGScale extension import: OK`. The build must be run
on the target machine and produces an ignored local `.so` file. The CUDA 11.6
environment includes compatible Conda GCC/G++ compilers for CUDA sources. The
build script uses a system `g++` for the Python binding and final link. If
`nvcc` is missing, check that the Conda environment is activated. If CUDA is
unavailable, check the NVIDIA driver and GPU visibility before running the
benchmark.

## 3. Download the checkpoint and its source dataset

This repository includes the `T_upper` tables but **no pretrained models or
dataset images**. Download the [official 3D-GS pretrained models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)
and the [Tanks and Temples / Deep Blending COLMAP data](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip),
both linked in the [official 3D-GS README](https://github.com/graphdeco-inria/gaussian-splatting#readme).
Download Mip-NeRF 360 scenes from the [dataset authors' page](https://jonbarron.info/mipnerf360/)
for those nine scenes.
For this example, locate the `train` model and the Tanks and Temples `train`
dataset. They must contain:

```text
<scene model>/cameras.json
<scene model>/point_cloud/iteration_30000/point_cloud.ply
<dataset scene>/images/<RGB images>
<dataset scene>/sparse/0/cameras.bin and images.bin
AdaGScale_GitHub/tupper_tables/train_ori_small_16x16.csv
```

The source COLMAP files can be `.txt` instead of `.bin`. Official 3D-GS
checkpoints use `cameras.json`; `cameras_Flash.json` and `cameras_3DGS.json`
are files in the author's local workspace and are not required here. Set the
parent directories containing your downloaded scene folders:

```bash
export MODEL_ROOT="$HOME/data/models"
export DATA_ROOT="$HOME/data"
```

The script expects `$MODEL_ROOT/train` and `$DATA_ROOT/tandt/train`. It also
accepts `train_ori` as a model folder name. Replace these paths if you
extracted the downloads elsewhere. A
`T_upper` table is tied to its scene, checkpoint, image resolution, and tile
size. A separately trained checkpoint may need a different table; extraction
code is not distributed.

## 4. Prepare and check one scene

Pass the scene name to the shell script:

```bash
bash run_official_scene.sh train --validate-only
```

Expect 263 training and 38 test records for `train`. The script follows the
official every-eighth registered image split and writes
`local/official_3dgs/train/{train,test,calibration_16,config}.json`.
The 16 calibration cameras are the names recorded in the T_upper manifest.
Some official `cameras.json` files contain only test cameras; the helper
reconstructs missing training records from the dataset's `sparse/0`. A missing
COLMAP model cannot reconstruct training cameras from a test-only JSON.

For another scene, replace `train` with `truck`, `drjohnson`, `playroom`,
`bicycle`, `flowers`, `garden`, `stump`, `treehill`, `bonsai`, `counter`,
`kitchen`, or `room`. The script selects `images`, `images_4`, or `images_2`
according to the scene. See [dataset layout](DATA_LAYOUT.md) for all folders
and separate dataset root variables.

## 5. Run the experiment

```bash
bash run_official_scene.sh train --protocol both --target-drop 0.5
```

The script prepares and validates the selected scene before running it. For
example, replace `train` with `truck` to run Tanks and Temples `truck`.
Other `run_experiment.py` options can be passed after the scene name.

`actual` chooses K from 16 training views and then measures test FPS and test
PSNR drop. `oracle` chooses K from the test views and measures on those same
test views. Both use the included **training-derived 16-view `T_upper` table**.
`0.5` is a target PSNR decrease in dB, not the K value. The actual test PSNR
decrease can differ from the calibration target.

## 6. Read the results

Each protocol writes a separate directory under
`results/train_ori_small_16x16/`. In its `summary.json`, check
`k_selection_split`, `k`, `kernel_input_param`, `calibration_drop_db`, `measured_test_drop_db`,
`baseline_test_fps`, `adagscale_test_fps`, and `speedup_vs_flashgs`.
`benchmark.json` contains per-trial FPS, pair counts, and PSNR. The
`results/` directory is ignored by Git.

The script chooses the matching `tupper_tables/*.csv` and saves its config
under `local/official_3dgs/<scene>/config.json`. These generated files and
results are ignored by Git. Exact T_upper training-view names are recorded
in `tupper_tables/MANIFEST.csv`.
