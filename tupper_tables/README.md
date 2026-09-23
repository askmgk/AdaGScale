# Precomputed scene-specific T_upper tables

This directory contains 17 precomputed tables:

- 13 Original 3D-GS checkpoints: `*_ori_small_16x16.csv`;
- four CityGaussian checkpoints: `*_citygs_large_32x32.csv`.

Each table was computed from 16 training views. `MANIFEST.csv` records the
exact image names and source indices used for each table.

## Compatibility

A `T_upper` table is specific to its scene, checkpoint, camera selection,
image resolution, and renderer tile size. Use a supplied table only with the
corresponding checkpoint and configuration. A custom or independently trained
checkpoint requires a newly computed table; the extraction implementation is
not included in this release.

The `*_small_16x16.csv` tables require `"tile_size": 16`. The
`*_citygs_large_32x32.csv` tables require `"tile_size": 32`.

For the 13 Original 3D-GS scenes, `run_official_scene.sh <scene>` selects the
matching table and creates camera lists using the image names in
`MANIFEST.csv`.

## CSV format

Each table contains 20 depth bins over `[0,100]`, with these columns:

```text
bin_idx,depth_min,depth_max,Tupper
```

`Tupper` is in `(0,1]`. A blank value is loaded as `1.0`, providing a
conservative bound for an empty bin.

Both the `actual` and `oracle` protocols use the same training-derived
`T_upper` table. Only the views used to select `K` differ between the two
protocols.
