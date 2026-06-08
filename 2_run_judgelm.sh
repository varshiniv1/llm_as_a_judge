#!/bin/bash
# ============================================================
#  SLURM job — JudgeLM baseline via vLLM
#  Runs forward pairs (scoring) + reverse pairs (positional bias)
#  Submit: sbatch 2_run_judgelm.sh
# ============================================================
#SBATCH --job-name=judgelm_vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/judgelm_%j.log
#SBATCH --error=logs/judgelm_%j.err

# ============================================================
# CONFIG — edit these
# ============================================================
WORK_DIR="$HOME/judgelm_baseline"
MODEL="BAAI/JudgeLM-7B-v1.0"          # or JudgeLM-13B-v1.0
CONDA_ENV="judgelm"
EXPECTED_ROWS=529                       # rows per pair file
MAX_NEW_TOKENS=512
HF_HOME="/work/$USER/hf_cache"          # avoid home-dir quota

# ============================================================
# SETUP
# ============================================================
export HF_HOME
mkdir -p logs

# Load conda — adjust to your Unity module setup
module purge
module load conda/latest        # Unity module name; check with: module avail conda
conda activate "$CONDA_ENV"

# Install vLLM if not already present (safe no-op if installed)
pip install -q vllm

echo "=============================="
echo " JudgeLM vLLM Baseline"
echo " Model:    $MODEL"
echo " WORK_DIR: $WORK_DIR"
echo " GPU:      $CUDA_VISIBLE_DEVICES"
echo "=============================="
echo ""

# ============================================================
# FORWARD PAIRS  (for scoring / ranking)
# ============================================================
echo "--- FORWARD PAIRS ---"
python run_judgelm_vllm.py \
    --input_dir      "$WORK_DIR/data/pairs/forward" \
    --output_dir     "$WORK_DIR/outputs/pairwise/forward" \
    --model          "$MODEL" \
    --expected_rows  "$EXPECTED_ROWS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature    0.0 \
    --tensor_parallel 1

echo ""

# ============================================================
# REVERSE PAIRS  (for positional bias test)
# ============================================================
echo "--- REVERSE PAIRS (positional bias) ---"
python run_judgelm_vllm.py \
    --input_dir      "$WORK_DIR/data/pairs/reverse" \
    --output_dir     "$WORK_DIR/outputs/pairwise/reverse" \
    --model          "$MODEL" \
    --expected_rows  "$EXPECTED_ROWS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature    0.0 \
    --tensor_parallel 1

echo ""
echo "=============================="
echo " Done. Run 3_aggregate_eval.py"
echo "=============================="
