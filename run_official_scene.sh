#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'EOF'
Usage: MODEL_ROOT=/path/to/models DATA_ROOT=/path/to/data bash run_official_scene.sh SCENE [run_experiment.py options]

Scenes: train truck drjohnson playroom bicycle flowers garden stump treehill bonsai counter kitchen room
Examples:
  MODEL_ROOT=/path/to/models DATA_ROOT=/path/to/data bash run_official_scene.sh train --validate-only
  MODEL_ROOT=/path/to/models DATA_ROOT=/path/to/data bash run_official_scene.sh truck --protocol both --target-drop 0.5

DATA_ROOT contains tandt/, db/, and the nine Mip-NeRF 360 scene folders.
Set TANDT_ROOT, DB_ROOT, or MIPNERF360_ROOT if those groups are stored elsewhere.
EOF
}

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

scene=$1
shift
case "$scene" in
  train|truck)
    group_root=${TANDT_ROOT:-${DATA_ROOT:+$DATA_ROOT/tandt}}
    image_folder=images
    group_label=TANDT_ROOT
    ;;
  drjohnson|playroom)
    group_root=${DB_ROOT:-${DATA_ROOT:+$DATA_ROOT/db}}
    image_folder=images
    group_label=DB_ROOT
    ;;
  bicycle|flowers|garden|stump|treehill)
    group_root=${MIPNERF360_ROOT:-${DATA_ROOT:-}}
    image_folder=images_4
    group_label=MIPNERF360_ROOT
    ;;
  bonsai|counter|kitchen|room)
    group_root=${MIPNERF360_ROOT:-${DATA_ROOT:-}}
    image_folder=images_2
    group_label=MIPNERF360_ROOT
    ;;
  *)
    printf 'Unknown scene: %s\n' "$scene" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "${MODEL_ROOT:-}" ]]; then
  printf 'Set MODEL_ROOT to the directory containing the official scene models.\n' >&2
  exit 2
fi
if [[ -z "$group_root" ]]; then
  printf 'Set DATA_ROOT or %s to the directory containing %s.\n' "$group_label" "$scene" >&2
  exit 2
fi

model_dir=$MODEL_ROOT/$scene
if [[ ! -f "$model_dir/cameras.json" || ! -f "$model_dir/point_cloud/iteration_30000/point_cloud.ply" ]]; then
  model_dir=$MODEL_ROOT/${scene}_ori
fi
if [[ ! -f "$model_dir/cameras.json" || ! -f "$model_dir/point_cloud/iteration_30000/point_cloud.ply" ]]; then
  printf 'Missing official model: expected %s/{%s,%s_ori}/cameras.json and point_cloud/iteration_30000/point_cloud.ply\n' "$MODEL_ROOT" "$scene" "$scene" >&2
  exit 2
fi

images_dir=$group_root/$scene/$image_folder
if [[ ! -d "$images_dir" ]]; then
  printf 'Missing RGB images: %s\n' "$images_dir" >&2
  exit 2
fi

split_root=${SPLIT_ROOT:-$repo_root/local/official_3dgs}
output_dir=$split_root/$scene
python "$repo_root/prepare_official_3dgs.py" \
  --model-dir "$model_dir" \
  --images-dir "$images_dir" \
  --output-dir "$output_dir" \
  --scene "$scene"

config=$output_dir/config.json
python "$repo_root/run_experiment.py" --config "$config" --validate-only
for option in "$@"; do
  if [[ "$option" == "--validate-only" ]]; then
    exit 0
  fi
done
python "$repo_root/run_experiment.py" --config "$config" "$@"
