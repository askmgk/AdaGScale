# Dataset and checkpoint layout

Dataset downloads and pretrained checkpoints are not part of this repository.
The first section covers official Original 3D-GS scenes. The second covers
other checkpoints and custom path layouts.

## Official Original 3D-GS models and datasets

The [official 3D-GS repository](https://github.com/graphdeco-inria/gaussian-splatting#readme)
links pretrained models, Tanks and Temples / Deep Blending COLMAP data, and
Mip-NeRF 360 source data. For each scene, identify the model directory by its
`cameras.json` and `point_cloud/iteration_30000/point_cloud.ply`. A matching
COLMAP source dataset normally has this structure:

```text
<scene model>/
  cameras.json
  point_cloud/iteration_30000/point_cloud.ply
<scene dataset>/
  sparse/0/cameras.bin
  sparse/0/images.bin
  images/                       Tanks and Temples / Deep Blending
  images_2/ or images_4/        Mip-NeRF 360 at model resolution
```

The sparse camera/image files may be `.txt` instead of `.bin`. Choose the
image folder matching `cameras.json` resolution. The official 3D-GS COLMAP
evaluation split holds out every eighth registered image in sorted name
order. An official model's `cameras.json` may list all views or only held-out
views; for the latter, the dataset's `sparse/0` is needed to reconstruct
training cameras.

Point `MODEL_ROOT` to the parent directory containing the official model
folders. Point `DATA_ROOT` to a directory with this layout:

```text
DATA_ROOT/
  tandt/{train,truck}/{images,sparse/0}/
  db/{drjohnson,playroom}/{images,sparse/0}/
  {bicycle,flowers,garden,stump,treehill}/{images_4,sparse/0}/
  {bonsai,counter,kitchen,room}/{images_2,sparse/0}/
```

The braces above list alternatives; each scene has its own directory. If
the three dataset groups have different parents, set `TANDT_ROOT`, `DB_ROOT`,
and `MIPNERF360_ROOT` to the directories directly containing their scenes.
The shell script selects the appropriate image folder for each scene:

```bash
export MODEL_ROOT=/path/to/models
export DATA_ROOT=/path/to/data
bash run_official_scene.sh train --validate-only
bash run_official_scene.sh truck --protocol both --target-drop 0.5
bash run_official_scene.sh bicycle --protocol both --target-drop 0.5
```

The script supports all 13 original 3D-GS evaluation scenes: `train`,
`truck`, `drjohnson`, `playroom`, `bicycle`, `flowers`, `garden`, `stump`,
`treehill`, `bonsai`, `counter`, `kitchen`, and `room`. It accepts model
folders named `<scene>` or `<scene>_ori`, writes the selected scene's camera
lists and config to `local/official_3dgs/<scene>/`, and validates them before
running. The provided T_upper table and `calibration_16.json` match the 16
training views listed in `tupper_tables/MANIFEST.csv`.

If the COLMAP directory is not a sibling of the image directory, use
`prepare_official_3dgs.py --model-dir ... --images-dir ... --colmap-dir ...
--output-dir ... --scene <name>` and then run the generated `config.json` with
`run_experiment.py`. The helper checks model and source camera geometry and
image resolution; it does not resize images. A different checkpoint or image
resolution may require a new T_upper table.

## Other pretrained checkpoints

The repository also supplies T_upper tables for GaussianSpa and CityGaussian
checkpoints. These need their own matching PLY files, camera metadata, and
images; the official Original 3D-GS scene script does not select them. Use
the table named for the checkpoint and renderer tile size. The
[`MANIFEST.csv`](../tupper_tables/MANIFEST.csv) file records the 16 views used
to extract each table.

For the supplied CityGaussian large checkpoints (`building`, `rubble`,
`residence`, and `sciart`), use the matching `*_citygs_large_32x32.csv` table
and set `"tile_size": 32` in the scene config. This selects the 32×32
preprocessing and `render_32x32` path. The supplied Original 3D-GS and
GaussianSpa small tables use `"tile_size": 16`.

A checkpoint can provide one complete camera JSON, separate train/test JSONs,
or a test JSON that is a subset of an all-camera JSON. When the camera lists
overlap, create disjoint lists with `prepare_camera_splits.py`:

```bash
python prepare_camera_splits.py \
  --all-cameras /path/to/all_cameras.json \
  --test-cameras /path/to/test_cameras.json \
  --output-dir local/splits/my_scene
```

If the source JSON orders all test views first and there is no separate test
JSON, use `--test-prefix-count N` with the known test-view count. If the
source follows a documented every-Nth holdout split, use `--test-stride N`.
Do not guess a split from filenames. `run_experiment.py` checks that the
configured train and test lists do not overlap.

For a custom checkpoint, write a scene config JSON with these fields:

- `case_id`: a unique scene/model/resolution name;
- `model_ply`: the matching checkpoint PLY;
- `tupper_csv`: the matching supplied table;
- `tile_size`: 16 or 32, matching the table;
- `train.cameras_json` and `train.gt_dir`: training cameras and RGB images;
- `test.cameras_json` and `test.gt_dir`: held-out cameras and RGB images.

You may set `train.calibration_cameras_json` to an exact set of 16 training
views. Otherwise the runner selects 16 at a fixed interval from the training
list. The oracle run uses every test camera to choose K. Both protocols
measure FPS and PSNR on the test cameras.

```bash
python run_experiment.py --config /path/to/my_scene.json --validate-only
python run_experiment.py --config /path/to/my_scene.json --protocol both --target-drop 0.5
```

The first command checks file paths, camera names, image sizes, train/test
separation, and T_upper format without rendering. Results go under `results/`.
