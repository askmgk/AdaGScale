# Dataset and checkpoint layout

Checkpoints and dataset images are not included in this repository.

## Official Original 3D-GS checkpoints

Download the checkpoints and source datasets linked by the
[official 3D-GS repository](https://github.com/graphdeco-inria/gaussian-splatting#readme).
Each scene requires the following files:

```text
<scene model>/
  cameras.json
  point_cloud/iteration_30000/point_cloud.ply

<scene dataset>/
  sparse/0/cameras.bin
  sparse/0/images.bin
  images/                       Tanks and Temples / Deep Blending
  images_2/ or images_4/        Mip-NeRF 360
```

The COLMAP files under `sparse/0` may be `.txt` instead of `.bin`. Use the
image directory whose resolution matches `cameras.json`.

Set `MODEL_ROOT` to the directory containing the model scene folders. The
script accepts a model folder named either `<scene>` or `<scene>_ori`.

Set `DATA_ROOT` to a directory with this structure:

```text
DATA_ROOT/
  tandt/
    train/{images,sparse/0}/
    truck/{images,sparse/0}/
  db/
    drjohnson/{images,sparse/0}/
    playroom/{images,sparse/0}/
  bicycle/{images_4,sparse/0}/
  flowers/{images_4,sparse/0}/
  garden/{images_4,sparse/0}/
  stump/{images_4,sparse/0}/
  treehill/{images_4,sparse/0}/
  bonsai/{images_2,sparse/0}/
  counter/{images_2,sparse/0}/
  kitchen/{images_2,sparse/0}/
  room/{images_2,sparse/0}/
```

Example:

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
bash run_official_scene.sh train --validate-only
bash run_official_scene.sh truck --protocol both --target-drop 0.5
bash run_official_scene.sh bicycle --protocol both --target-drop 0.5
```

If the dataset groups have different parent directories, set the corresponding
variables instead of `DATA_ROOT`:

```bash
export TANDT_ROOT=/path/containing/train_and_truck
export DB_ROOT=/path/containing/drjohnson_and_playroom
export MIPNERF360_ROOT=/path/containing/mipnerf360_scenes
```

The preparation script follows the official every-eighth-image test split. If
an official `cameras.json` contains only the test views, it reconstructs the
training camera records from the matching COLMAP model under `sparse/0`.

If `sparse/0` is not next to the image directory, run the preparation script
directly:

```bash
python prepare_official_3dgs.py \
  --model-dir /path/to/models/train \
  --images-dir /path/to/data/tandt/train/images \
  --colmap-dir /path/to/sparse/0 \
  --output-dir local/official_3dgs/train \
  --scene train
```

## CityGaussian tables

The repository also supplies precomputed `T_upper` tables for four CityGaussian
checkpoints. These tables require the matching checkpoint, camera metadata,
images, resolution, and tile size; the Original 3D-GS scene script does not
select them automatically.

The CityGaussian tables for `building`, `rubble`, `residence`, and `sciart`
require `"tile_size": 32`. The supplied Original 3D-GS tables use
`"tile_size": 16`. See the
[`T_upper` table documentation](../tupper_tables/README.md) for file naming and
compatibility requirements.

## Custom checkpoints

`T_upper` depends on the checkpoint, selected cameras, image resolution, and
renderer tile size. A custom or independently trained checkpoint therefore
requires a newly computed `T_upper` table. The extraction implementation is not
included in this release.

After obtaining a matching table, create a scene configuration with these
fields:

- `case_id`: a unique scene/checkpoint/resolution name;
- `model_ply`: the checkpoint PLY path;
- `tupper_csv`: the matching `T_upper` CSV path;
- `tile_size`: `16` or `32`, matching the table;
- `train.cameras_json` and `train.gt_dir`: training cameras and images;
- `test.cameras_json` and `test.gt_dir`: held-out cameras and images;
- `train.calibration_cameras_json`: optional explicit calibration-view list.

Validate and run the configuration as follows:

```bash
python run_experiment.py --config /path/to/config.json --validate-only
python run_experiment.py --config /path/to/config.json --protocol both --target-drop 0.5
```

If a checkpoint provides overlapping all-view and test-view camera files, make
disjoint lists with:

```bash
python prepare_camera_splits.py \
  --all-cameras /path/to/all_cameras.json \
  --test-cameras /path/to/test_cameras.json \
  --output-dir local/splits/my_scene
```

`prepare_camera_splits.py --help` lists options for a known test prefix or a
documented every-Nth holdout split. Do not infer a split from filenames.
