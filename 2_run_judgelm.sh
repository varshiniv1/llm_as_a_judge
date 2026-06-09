#!/bin/bash
# ============================================================
#  SLURM job — JudgeLM baseline via HuggingFace Transformers
#  Runs forward pairs (scoring) + reverse pairs (positional bias)
#  Submit from ~/llm_as_a_judge:  sbatch 2_run_judgelm.sh
# ============================================================
#SBATCH --job-name=judgelm_hf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/llm_as_a_judge/logs/judgelm_%j.log
#SBATCH --error=/home/%u/llm_as_a_judge/logs/judgelm_%j.err

set -e

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
BATCH_SIZE=8         # safe for A100 40GB with 7B model in float16

export HF_HOME="/work/pi_dagarwal_umass_edu/$USER/hf_cache"

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
# ENSURE CONDA ENV IS VALID (recreate if broken)
# ============================================================
ENV_PATH="/work/pi_dagarwal_umass_edu/${USER}/.conda/envs/${CONDA_ENV}"

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
if ! conda run -n "$CONDA_ENV" python -c "import transformers, torch" 2>/dev/null; then
    echo "--- Installing Python dependencies ---"
    conda run -n "$CONDA_ENV" pip install \
        torch torchvision torchaudio \
        transformers accelerate \
        pandas numpy scipy scikit-learn
    echo "--- Install complete ---"
fi

conda run -n "$CONDA_ENV" python -c \
    "import transformers, torch, pandas, scipy, sklearn; \
     print('Deps OK | torch', torch.__version__, '| transformers', transformers.__version__)"

echo "=============================="
echo " JudgeLM HF Baseline"
echo " User:     $USER"
echo " Repo:     $REPO_DIR"
echo " WORK_DIR: $WORK_DIR"
echo " HF_HOME:  $HF_HOME"
echo " GPU:      $CUDA_VISIBLE_DEVICES"
echo " Batch:    $BATCH_SIZE"
echo "=============================="
echo ""

# ============================================================
# STEP 1 — Prepare pairwise JSONL files
# ============================================================
echo "--- PREPARING PAIRS ---"
conda run -n "$CONDA_ENV" python 1_prepare_pairs.py
echo ""

# ============================================================
# FORWARD PAIRS
# ============================================================
echo "--- FORWARD PAIRS ---"
conda run -n "$CONDA_ENV" python run_judgelm_hf.py \
    --input_dir      "$WORK_DIR/data/pairs/forward" \
    --output_dir     "$WORK_DIR/outputs/pairwise/forward" \
    --model          "$MODEL" \
    --expected_rows  "$EXPECTED_ROWS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size     "$BATCH_SIZE"

echo ""

# ============================================================
# REVERSE PAIRS
# ============================================================
echo "--- REVERSE PAIRS (positional bias) ---"
conda run -n "$CONDA_ENV" python run_judgelm_hf.py \
    --input_dir      "$WORK_DIR/data/pairs/reverse" \
    --output_dir     "$WORK_DIR/outputs/pairwise/reverse" \
    --model          "$MODEL" \
    --expected_rows  "$EXPECTED_ROWS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size     "$BATCH_SIZE"

echo ""
echo "=============================="
echo " Done. Run: python 3_aggregate_eval.py"
echo "=============================="
