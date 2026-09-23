# Source provenance

The CUDA renderer and Python extension in `csrc/` and `setup.py` are based on
[FlashGS](https://github.com/InternLandMark/FlashGS) and include the AdaGScale
depth-LUT preprocessing kernel. The FlashGS copyright and MIT license are
preserved in `third_party/FlashGS_LICENSE`.

`csrc/glm/` contains vendored [OpenGL Mathematics (GLM)](https://github.com/g-truc/glm)
headers. GLM offers a choice of the Happy Bunny License or MIT License; this
release uses the MIT option. Its notice is in `third_party/GLM_LICENSE`.

The K calibration and benchmark scripts and the supplied scene-specific
`T_upper` CSV tables were developed for AdaGScale. The `T_upper` extraction
implementation, pretrained checkpoints, dataset images, and raw experiment
outputs are not included.
