#!/bin/bash
# ============================================================
#  SLURM job — JudgeLM baseline via vLLM
#  Runs forward pairs (scoring) + reverse pairs (positional bias)
#  Submit from ~/llm_as_a_judge:  sbatch 2_run_judgelm.sh
# ============================================================
#SBATCH --job-name=judgelm_vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/llm_as_a_judge/logs/judgelm_%j.log
#SBATCH --error=/home/%u/llm_as_a_judge/logs/judgelm_%j.err

# ============================================================
# Always run from inside the cloned repo
# ============================================================
cd "$HOME/llm_as_a_judge"

# ============================================================
# CONFIG
# ============================================================
REPO_DIR="$HOME/llm_as_a_judge"
WORK_DIR="$REPO_DIR/judgelm_baseline"   # pairs + outputs live here
MODEL="BAAI/JudgeLM-7B-v1.0"
CONDA_ENV="judgelm"
EXPECTED_ROWS=529
MAX_NEW_TOKENS=512

# HuggingFace cache — keeps model weights off the home quota.
HF_HOME="/work/pi_dagarwal_umass_edu/$USER/hf_cache"

# ============================================================
# SETUP
# ============================================================
export HF_HOME
mkdir -p "$REPO_DIR/logs"
mkdir -p "$WORK_DIR/outputs/pairwise/forward"
mkdir -p "$WORK_DIR/outputs/pairwise/reverse"

module purge
module load conda/latest
conda activate "$CONDA_ENV"

echo "=============================="
echo " JudgeLM vLLM Baseline"
echo " User:     $USER"
echo " Repo:     $REPO_DIR"
echo " WORK_DIR: $WORK_DIR"
echo " HF_HOME:  $HF_HOME"
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
echo " Done. Run: python 3_aggregate_eval.py"
echo "=============================="
