# Precomputed scene-specific T_upper tables

This directory contains 30 scene-specific CSV tables: 13 Original 3D-GS,
13 GaussianSpa, and four CityGaussian checkpoints. Each table was
**re-extracted on an NVIDIA RTX A6000 from 16 training camera views** using
the author's local CUDA extraction implementation. The implementation and
per-view intermediate data are not distributed here.

`MANIFEST.csv` records the source camera JSON filename, the exact 16 indices
in that JSON, and their image names for every table. Indices and image names
are semicolon-separated. The starting camera index follows the author's
original train/test ordering for each scene. The table filename identifies
the scene, checkpoint type, resolution, and renderer tile size. Use only the
CSV that matches the checkpoint and tile size in the scene config.
The four `*_citygs_large_32x32.csv` tables require `"tile_size": 32`; the
`*_small_16x16.csv` tables require `"tile_size": 16`.

The supplied tables now match the AdaGScale paper's stated **16-training-view
count**. Exact numeric reproduction also depends on the checkpoint, camera
ordering, renderer version, and selected views; use `MANIFEST.csv` to report
these choices. Both actual and oracle K protocols use the same training-derived
`T_upper` table. Oracle tuning changes only which views are used to select K.

For the 13 Original 3D-GS scenes, all 16 listed source images fall in the
official every-eighth-image training split. The official pretrained model uses
`cameras.json`; `cameras_Flash.json` in the manifest records the local source
file used for table extraction. Run `bash run_official_scene.sh <scene>` to
select the corresponding CSV and prepare matching train/test camera JSONs.

The CSV fields are `bin_idx,depth_min,depth_max,Tupper`.
There are 20 depth bins, each five units wide, spanning `[0,100]`.
`Tupper` is a number in `(0,1]`; a blank value is interpreted as `1.0` by
`tupper.py` for a conservative empty-bin bound.
