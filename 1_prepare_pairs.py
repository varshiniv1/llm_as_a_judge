"""
Step 1 — Prepare pairwise JSONL input files for JudgeLM inference.

Produces:
  judgelm_baseline/data/pairs/forward/   6 pair files (for scoring)
  judgelm_baseline/data/pairs/reverse/   6 pair files (order-flipped, for positional bias)

Each file has 529 rows (one per question).

WHAT THIS SCRIPT DOES:
  1. Load human_annotated_dataset.csv (produced by 0_prepare_data.py)
  2. Truncate responses that are too long to fit inside JudgeLM-7B's context
     window (4096 tokens total).  See MAX_RESPONSE_WORDS below.
  3. Write 12 JSONL files: 6 forward + 6 reverse orderings.
     Each row records which models were in which position, the truncation
     flag, and an estimated token count so you can cross-check after inference.

WHY RESPONSE TRUNCATION:
  JudgeLM-7B-v1.0 has a 4096-token context window.
  With max_new_tokens=512, the available input budget is 3584 tokens
  (approximately 2654 words at 1.35 tok/word).

  The dataset contains an extreme outlier: row 503 meta-llama response
  = 2854 words (~3853 tokens), which alone exceeds the entire context.
  Any pair involving that response would be silently truncated by vLLM
  mid-sentence, producing an incoherent input and an unreliable judgment.

  Instead we apply clean, deterministic truncation here:
    - Limit every response to MAX_RESPONSE_WORDS words (first N words).
    - Append the literal marker '[TRUNCATED]' so the model and downstream
      scripts can detect shortened text.
    - The truncation is per-response; the marker travels into both forward
      and reverse files.

  MAX_RESPONSE_WORDS = 500 guarantees that even with the longest question
  (656 words) and reference answer (624 words) on the same row, the full
  prompt fits within the 3584-word input budget:
      question(656) + reference(624) + 2*response(500) + template(80)
      = 2360 words approx 3186 tokens  < 3584 token budget  [OK]

WHY FORWARD + REVERSE:
  Positional bias test: if JudgeLM prefers Answer-1 simply because it
  appears first (not because it is better), swapping the order will flip the
  winner.  Running every pair in both orders and measuring agreement rate
  (consistency) and position-1 preference rate quantifies this bias.

MODEL_ORDER is fixed alphabetically so pair file names are deterministic
and match the column numbering in 3_aggregate_eval.py.

Run on Unity AFTER uploading human_annotated_dataset.csv:
  python 1_prepare_pairs.py
"""

import json
import itertools
import statistics
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "human_annotated_dataset.csv"
WORK_DIR = Path("judgelm_baseline")

FWD_DIR  = WORK_DIR / "data" / "pairs" / "forward"
REV_DIR  = WORK_DIR / "data" / "pairs" / "reverse"
FWD_DIR.mkdir(parents=True, exist_ok=True)
REV_DIR.mkdir(parents=True, exist_ok=True)

# Maximum words per response before truncation.
# Derivation:
#   Input budget (4096 - 512 output tokens) / 1.35 tok/word approx 2654 words
#   Subtract worst-case question(656) + reference(624) + template(80) = 1360 words
#   Remaining for 2 responses approx 1294 words -> 647 per response
#   Use 500 for a comfortable safety margin.
MAX_RESPONSE_WORDS = 500

# Short alias -> CSV column name  (must match 3_aggregate_eval.py)
MODEL_COLS = {
    "gemini_flash": "google/gemini-2.5-flash_response",
    "qwen3_32b":    "qwen/qwen3-32b_response",
    "claude_haiku": "anthropic/claude-3-haiku_response",
    "llama3_2_1b":  "meta-llama/llama-3.2-1b-instruct_response",
}

# Alphabetical -- matches 3_aggregate_eval.py MODEL_ORDER
MODEL_ORDER = sorted(MODEL_COLS.keys())
# ['claude_haiku', 'gemini_flash', 'llama3_2_1b', 'qwen3_32b']

PAIRS = list(itertools.combinations(MODEL_ORDER, 2))   # 6 pairs C(4,2)


# ============================================================
# HELPERS
# ============================================================
def truncate_response(text: str, max_words: int = MAX_RESPONSE_WORDS):
    """
    Truncate text to at most max_words words.
    Returns (text_out, was_truncated).
    Appends '[TRUNCATED]' marker if truncation occurred.
    """
    if not text or str(text).strip() == "":
        return str(text), False
    words = str(text).split()
    if len(words) <= max_words:
        return str(text), False
    return " ".join(words[:max_words]) + " [TRUNCATED]", True


def estimate_prompt_words(question, reference, answer1, answer2):
    """
    Estimate total word count of a JudgeLM prompt.
    Template overhead (section headers + system message) approx 80 words.
    """
    template_overhead = 80
    return (
        len(str(question).split())
        + len(str(reference).split())
        + len(str(answer1).split())
        + len(str(answer2).split())
        + template_overhead
    )


# ============================================================
# LOAD
# ============================================================
df = pd.read_csv(CSV_PATH)
N  = len(df)

print("=" * 65)
print(" Step 1 -- Prepare Pairwise JSONL Files")
print("=" * 65)
print(f"\nLoaded {N} rows from {CSV_PATH}")
print(f"Model order (alphabetical): {MODEL_ORDER}")
print(f"Pairs to generate: {len(PAIRS)}")
print(f"Max response words (truncation threshold): {MAX_RESPONSE_WORDS}")

# ============================================================
# PRE-PROCESS RESPONSES (truncation)
# ============================================================
print("\n--- Pre-processing responses (truncation check) ---")
truncated_log = {}   # {model_alias: [(row_idx, original_len), ...]}

processed = {}   # {model_alias: [(text, was_truncated), ...]}
for alias, col in MODEL_COLS.items():
    texts      = []
    trunc_list = []
    for i, val in enumerate(df[col]):
        text, was_truncated = truncate_response(str(val) if pd.notna(val) else "")
        texts.append((text, was_truncated))
        if was_truncated:
            orig_len = len(str(val).split())
            trunc_list.append((i, orig_len))
    processed[alias] = texts
    if trunc_list:
        truncated_log[alias] = trunc_list
        print(f"  {alias}: truncated {len(trunc_list)} response(s)")
        for row_i, orig_len in trunc_list:
            print(f"    row {row_i}: {orig_len} words -> {MAX_RESPONSE_WORDS} words [TRUNCATED]")
    else:
        print(f"  {alias}: no truncation needed")

total_truncated_cells = sum(len(v) for v in truncated_log.values())
print(f"\n  Total cells truncated: {total_truncated_cells}")

# ============================================================
# WRITE JSONL FILES
# ============================================================
print("\n--- Writing pair files ---")

all_token_estimates = []   # accumulate for budget summary

for model_a, model_b in PAIRS:
    pair_name = f"{model_a}__vs__{model_b}"
    fwd_rows  = []
    rev_rows  = []

    for i, row in df.iterrows():
        q      = str(row["question"])
        ref    = str(row["original_answer"])
        a_text, a_trunc = processed[model_a][i]
        b_text, b_trunc = processed[model_b][i]

        est_words  = estimate_prompt_words(q, ref, a_text, b_text)
        est_tokens = int(est_words * 1.35)
        all_token_estimates.append(est_tokens)

        base = {
            "question_id":           int(i),
            "question_body":         q,
            "reference":             ref,
            "prompt_word_estimate":  est_words,
            "prompt_token_estimate": est_tokens,
            "truncation_applied":    a_trunc or b_trunc,
        }
        fwd_rows.append({
            **base,
            "answer1_body":      a_text,
            "answer2_body":      b_text,
            "answer1_model_id":  model_a,
            "answer2_model_id":  model_b,
            "answer1_truncated": a_trunc,
            "answer2_truncated": b_trunc,
            "reverse":           False,
        })
        rev_rows.append({
            **base,
            # Swap A and B for positional bias test
            "answer1_body":      b_text,
            "answer2_body":      a_text,
            "answer1_model_id":  model_b,
            "answer2_model_id":  model_a,
            "answer1_truncated": b_trunc,
            "answer2_truncated": a_trunc,
            "reverse":           True,
        })

    fwd_path = FWD_DIR / f"{pair_name}.jsonl"
    rev_path = REV_DIR / f"{pair_name}.jsonl"

    with open(fwd_path, "w", encoding="utf-8") as f:
        for r in fwd_rows:
            f.write(json.dumps(r) + "\n")

    with open(rev_path, "w", encoding="utf-8") as f:
        for r in rev_rows:
            f.write(json.dumps(r) + "\n")

    print(f"  {pair_name}: {len(fwd_rows)} rows  -> forward + reverse")

# ============================================================
# TOKEN BUDGET SUMMARY
# ============================================================
print(f"\n--- Token budget summary (all {len(all_token_estimates)} forward pair rows) ---")
sorted_estimates = sorted(all_token_estimates)
p95_idx = int(0.95 * len(sorted_estimates))
print(f"  min    : {min(all_token_estimates)}")
print(f"  median : {int(statistics.median(all_token_estimates))}")
print(f"  p95    : {sorted_estimates[p95_idx]}")
print(f"  max    : {max(all_token_estimates)}")
print(f"  JudgeLM effective input budget: ~3584 tokens (4096 - 512 output)")

over_budget = [t for t in all_token_estimates if t > 3584]
if over_budget:
    print(f"  [WARN] {len(over_budget)} pair rows still estimated over budget.")
    print(f"         Consider reducing MAX_RESPONSE_WORDS (currently {MAX_RESPONSE_WORDS}).")
else:
    print(f"  [OK]  All pair rows fit within context window.")

# ============================================================
# FINAL REPORT
# ============================================================
fwd_count = len(list(FWD_DIR.glob("*.jsonl")))
rev_count = len(list(REV_DIR.glob("*.jsonl")))

print(f"\n{'='*65}")
print(f"  Forward files : {fwd_count}/6  -> {FWD_DIR}")
print(f"  Reverse files : {rev_count}/6  -> {REV_DIR}")
print(f"  Rows per file : {N}")
print(f"  Total JSONL rows written: {(fwd_count + rev_count) * N}")
if total_truncated_cells:
    print(f"\n  Truncation summary:")
    for alias, entries in truncated_log.items():
        for row_i, orig_len in entries:
            print(f"    {alias} row {row_i}: {orig_len} -> {MAX_RESPONSE_WORDS} words")
print(f"\nDone. Submit 2_run_judgelm.sh next.")
