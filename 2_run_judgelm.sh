#!/bin/bash
# ============================================================
#  SLURM job — JudgeLM baseline via vLLM
#  Runs forward pairs (scoring) + reverse pairs (positional bias)
#  Submit from ~/llm_as_a_judge:  sbatch 2_run_judgelm.sh
# ============================================================
#SBATCH --job-name=judgelm_vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1          # V100 (sm_70) required — vLLM 0.22.1 needs cc>=7.0; GTX 1080 Ti (cc=6.1) is too old
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/llm_as_a_judge/logs/judgelm_%j.log
#SBATCH --error=/home/%u/llm_as_a_judge/logs/judgelm_%j.err

set -e   # exit immediately on any error

# ============================================================
# Always run from inside the cloned repo
# ============================================================
cd "$HOME/llm_as_a_judge"

# ============================================================
# CONFIG
# ============================================================
REPO_DIR="$HOME/llm_as_a_judge"
WORK_DIR="$REPO_DIR/judgelm_baseline"
MODEL="BAAI/JudgeLM-7B-v1.0"
CONDA_ENV="judgelm"
EXPECTED_ROWS=529
MAX_NEW_TOKENS=512

# HuggingFace cache — keeps model weights off the home quota.
export HF_HOME="/work/pi_dagarwal_umass_edu/$USER/hf_cache"

# JudgeLM-7B was fine-tuned for 4096 context but its config.json still
# reports max_position_embeddings=2048 (LLaMA-1 base). Newer vLLM
# rejects max_model_len > config value unless this flag is set.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Path to the conda env (first envs_dir from conda config)
ENV_PATH="/work/pi_dagarwal_umass_edu/${USER}/.conda/envs/${CONDA_ENV}"

# ============================================================
# SETUP
# ============================================================
mkdir -p "$REPO_DIR/logs"
mkdir -p "$WORK_DIR/outputs/pairwise/forward"
mkdir -p "$WORK_DIR/outputs/pairwise/reverse"
mkdir -p "$HF_HOME"

module purge
module load conda/latest

echo "Conda: $(conda --version)"

# ============================================================
# ENSURE CONDA ENV IS VALID
# If the directory exists but is not a real conda env
# (DirectoryNotACondaEnvironmentError), remove it and recreate.
# ============================================================
if ! conda run -n "$CONDA_ENV" python -c "print('env_ok')" 2>/dev/null; then
    echo "--- Conda env '$CONDA_ENV' missing or broken. Recreating... ---"
    rm -rf "$ENV_PATH"
    conda create -n "$CONDA_ENV" python=3.10 --yes
    echo "--- Env created ---"
fi

echo "Python: $(conda run -n $CONDA_ENV which python)"

# ============================================================
# INSTALL PYTHON DEPS (skipped if already present)
# ============================================================
if ! conda run -n "$CONDA_ENV" python -c "import vllm" 2>/dev/null; then
    echo "--- Installing Python dependencies ---"
    conda run -n "$CONDA_ENV" pip install vllm pandas numpy scipy scikit-learn
    echo "--- Install complete ---"
fi

# Hard verify — job dies here if anything is still missing
conda run -n "$CONDA_ENV" python -c \
    "import vllm, pandas, numpy, scipy, sklearn; print('Deps OK')"

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
# STEP 1 — Prepare pairwise JSONL files
# (safe to re-run: skips files that already exist at full size)
# ============================================================
echo "--- PREPARING PAIRS ---"
conda run -n "$CONDA_ENV" python 1_prepare_pairs.py
echo ""

# ============================================================
# FORWARD PAIRS  (for scoring / ranking)
# ============================================================
echo "--- FORWARD PAIRS ---"
conda run -n "$CONDA_ENV" python run_judgelm_vllm.py \
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
conda run -n "$CONDA_ENV" python run_judgelm_vllm.py \
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
