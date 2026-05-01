#!/bin/bash
#SBATCH --job-name=gen3r_re10k_exact
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --account=EUHPC_D30_012
#SBATCH --partition=boost_usr_prod
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONDA_BASE="/leonardo_work/EUHPC_D30_012/miniforge3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate gen3r

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

ROOT="/leonardo_scratch/fast/EUHPC_D30_012/re10k_preprocessed_subsampled_test"
OUT_ROOT="/leonardo_work/EUHPC_D30_012/results/gen3r_re10k_variant_exact"
CKPT_DIR="checkpoints"

mkdir -p logs
mkdir -p "${OUT_ROOT}"

# 1. Prepare exact target cameras and GT targets.
python tools/prepare_gen3r_re10k_exact.py \
  --root "${ROOT}" \
  --eval_json re10k_eval_v2.json \
  --out_root "${OUT_ROOT}/prepared" \
  --difficulty easy \
  --num_context_views 2 \
  --context_selection even \
  --max_samples 20 \
  --prompt "a realistic indoor room" \
  --color_dir_name color \
  --pose_dir_name pose \
  --intrinsic_dir_name intrinsic

# 2. Run Gen3R and extract predicted target RGB frames.
python tools/run_gen3r_exact_and_extract.py \
  --manifest "${OUT_ROOT}/prepared/manifest_exact.json" \
  --gen3r_root . \
  --checkpoint_dir "${CKPT_DIR}" \
  --limit 20 \
  --remove_far_points

# 3. Evaluate target GT vs predicted target RGB.
python tools/eval_gen3r_exact_metrics.py \
  --manifest "${OUT_ROOT}/prepared/manifest_exact_with_preds.json" \
  --out_json "${OUT_ROOT}/metrics.json" \
  --image_size 560 \
  --device cuda \
  --compute_lpips \
  --compute_fid