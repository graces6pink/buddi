#!/usr/bin/env bash
# Reproduce BUDDI paper Table 3 (CHI3D, pair s03) from datasets/original/CHI3D.
#
# Usage:
#   ./datasets/scripts/CHI3D/repro/run_all.sh prep          # steps 0-6 (CPU+GPU, ~1h)
#   ./datasets/scripts/CHI3D/repro/run_all.sh fit bev_init  # one Table 3 row
#   ./datasets/scripts/CHI3D/repro/run_all.sh fit buddi_gen
#   ./datasets/scripts/CHI3D/repro/run_all.sh fit buddi
#   ./datasets/scripts/CHI3D/repro/run_all.sh eval bev_init # print the metrics
#
# Every stage is idempotent: rerunning skips work whose output already exists.

set -euo pipefail

# the repo runs in the `buddi` conda env (not the hhcenv39 that install_conda_env.sh
# names); activate it here so the script works from any shell
CONDA_ENV="${CONDA_ENV:-buddi}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  # conda's activate.d hooks reference unbound vars, so relax -u around them
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  set -u
fi

cd "$(dirname "$0")/../../../.."
export PYTHONPATH="$PWD"
export ESSENTIALS_HOME="$PWD/essentials"   # chi3d_eval.py:28 reads this, unset by default
mkdir -p llib/methods/hhcs_optimization/evaluation/temp  # ResultLogger.topkl writes here

SUBJECTS="${SUBJECTS:-s03}"
BASE_FOLDER="${BASE_FOLDER:-demo/optimization/chi3d}"

# Which diffusion prior the `fit` rows use. Defaults reproduce the paper with the
# released weights (that is what RESULTS.md reports); override both to evaluate a
# checkpoint you trained, e.g.
#   CKPT=demo/diffusion/training/<run>/checkpoints/<x>.pt \
#   CKPT_CFG=demo/diffusion/training/<run>/config.yaml \
#   BASE_FOLDER=demo/optimization/chi3d_<run> ./run_all.sh fit buddi
#
# NOTE the cfg is only read for the model architecture and diffusion settings, but
# main.py:160-163 also takes its batch_size to size the SMPL-X buffers -- a training
# config says 512 while the fit runs one image at a time, which wastes ~8GB of VRAM.
# Pass a copy with batch_size: 1 if the GPU is shared. cfg/interx6_chi3d_ep323_bs1.yaml
# is exactly that: the interx6_chi3d_cond_bev training config with the one line changed.
# It is what RESULTS_interx6.md reports; see that file for the full pipeline.
CKPT="${CKPT:-essentials/buddi/buddi_cond_bev.pt}"
CKPT_CFG="${CKPT_CFG:-essentials/buddi/buddi_cond_bev.yaml}"
R=datasets/scripts/CHI3D/repro

case "${1:-}" in

prep)
  python $R/s00_fix_essentials.py
  python $R/s01_extract_contact_frames.py --subjects $SUBJECTS
  python $R/s02_run_vitpose.py            --subjects $SUBJECTS

  # BEV must run in video (folder) mode: it produces {name}__2_0.08.npz, the
  # double-underscore name process_chi3d.py:204 expects. The per-image form used
  # by demo.sh yields {name}_0.08.npz, which is silently never found.
  for s in $SUBJECTS; do
    d=datasets/processed/CHI3D/train/$s
    mkdir -p "$d/bev"
    bev -m video -i "$d/images_contact" -o "$d/bev"
  done

  python datasets/scripts/CHI3D/project_joints_chi3d.py
  python $R/s03_build_correspondence.py --subjects $SUBJECTS
  python $R/s04_process_chi3d.py        --subjects $SUBJECTS
  python $R/s05_make_split.py
  python $R/s06_sanity_check.py         --subjects $SUBJECTS
  ;;

fit)
  row="${2:?usage: run_all.sh fit bev_init|buddi_gen|buddi}"
  case "$row" in
    bev_init)  cfg=bev_init_chi3d.yaml      ;;
    buddi_gen) cfg=buddi_cond_bev_gen.yaml  ;;
    buddi)     cfg=buddi_cond_bev.yaml      ;;
    *) echo "unknown row: $row" >&2; exit 2  ;;
  esac

  python llib/methods/hhcs_optimization/main.py \
    --exp-cfg llib/methods/hhcs_optimization/configs/$cfg \
    --exp-opts \
      logging.base_folder=$BASE_FOLDER \
      logging.run=fit_${row}_chi3d_val \
      datasets.train_names=[] \
      datasets.train_composition=[] \
      datasets.val_names="['chi3d']" \
      datasets.test_names=[] \
      datasets.chi3d.image_folder=images_contact \
      model.optimization.pretrained_diffusion_model_ckpt="$CKPT" \
      model.optimization.pretrained_diffusion_model_cfg="$CKPT_CFG"
  ;;

eval)
  row="${2:?usage: run_all.sh eval bev_init|buddi_gen|buddi}"
  python llib/methods/hhcs_optimization/evaluation/chi3d_eval.py \
    --exp-cfg llib/methods/hhcs_optimization/evaluation/chi3d_eval.yaml \
    --predictions-folder $BASE_FOLDER/fit_${row}_chi3d_val \
    --eval-split val \
    --print_result
  echo
  echo "Table 3 columns:  PER PERSON = est_pa_mpjpe_h0 / est_pa_mpjpe_h1"
  echo "                  JOINT      = est_pa_mpjpe_h0h1"
  ;;

*)
  sed -n '2,12p' "$0"
  exit 2
  ;;
esac
