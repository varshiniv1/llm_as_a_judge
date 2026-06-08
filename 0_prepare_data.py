"""
Step 0 — Data Preparation for JudgeLM Baseline Pipeline
========================================================
Input : human_annotation_analysis_candidate_response_annotated.csv
Output: human_annotated_dataset.csv  (UTF-8, ready for Unity)

WHAT THIS SCRIPT DOES  (in order):
  Step 1  Audit raw annotations  — count coverage, find typos
  Step 2  Parse & clean Ranking columns
           - Fix double-comma typos  e.g. [2,1,3,,4] -> [2,1,3,4]
           - Validate each result is a permutation of [1,2,3,4]
           - Merge Varshini / Sanjna / Pramith into one Ranking column:
               1 annotator  -> use directly
               2-3 annotators -> average ranks per position, then re-rank
                                 (lowest average rank = rank 1)
  Step 3  Fill NaN response cells with ""
           (4 cells total: qwen rows 170,204,246; llama row 62)
  Step 4  Add question_id (0-indexed)
  Step 5  Validate final state
  Step 6  Save

WHY THESE CHOICES:
  - Double-comma fix: [2,1,3,,4] is clearly a typing error; correcting the
    formatting does not alter the annotator's intent. This recovers row 386's
    Sanjna annotation and brings all 529 rows to 100% coverage.
  - Permutation validation: ensures every accepted ranking uses each of
    {1,2,3,4} exactly once. Rankings like [1,1,3,4] would silently produce
    wrong Spearman/Kendall values downstream.
  - Consensus by averaged ranks (not majority vote): with non-overlapping
    annotators (only 3 rows have 2 annotators), averaging is equivalent to
    using the single annotator's value. For the 3 overlap rows it averages
    the two annotators' rank vectors and re-ranks — a standard aggregation
    used in information retrieval (Borda count variant).
  - NaN response cells -> "": JudgeLM still evaluates the pair; an empty
    response will score low, which reflects reality (the model produced
    nothing for those questions). Dropping the rows would lose 4 questions.
"""

import ast
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
SRC  = Path(r"C:\Users\varsh\Downloads\human_annotation_analysis_candidate_response_annotated.csv")
DEST = Path(r"C:\Users\varsh\Downloads\human_annotated_dataset.csv")

# ============================================================
# COLUMN DEFINITIONS
# ============================================================
RESPONSE_COLS = [
    "question",
    "original_answer",
    "google/gemini-2.5-flash_response",
    "qwen/qwen3-32b_response",
    "anthropic/claude-3-haiku_response",
    "meta-llama/llama-3.2-1b-instruct_response",
]

ANNOTATOR_COLS = ["Varshini", "Sanjna", "Pramith"]

OUTPUT_COL_ORDER = [
    "question_id",
    "question",
    "original_answer",
    "dataset_name",
    "google/gemini-2.5-flash_response",
    "qwen/qwen3-32b_response",
    "anthropic/claude-3-haiku_response",
    "meta-llama/llama-3.2-1b-instruct_response",
    "Ranking",
]

VALID_SET = [1, 2, 3, 4]   # every accepted ranking must be a permutation of this


# ============================================================
# HELPERS
# ============================================================
def normalize_annotation_str(s: str) -> str:
    """
    Fix common typos in annotation strings before parsing.
    Currently handles:
      - Two or more consecutive commas  e.g. '[2,1,3,,4]' -> '[2,1,3,4]'
      - Trailing/leading whitespace
    Returns the normalised string (may still be un-parseable).
    """
    s = s.strip()
    s = re.sub(r",{2,}", ",", s)     # collapse repeated commas
    return s


def parse_ranking(val) -> list | None:
    """
    Parse one annotation cell into a 4-element integer list, or None.

    Accepts:
      '[2, 1, 3, 4]'  -> [2, 1, 3, 4]   (standard Varshini/Pramith format)
      '[2,1,3,4]'     -> [2, 1, 3, 4]   (Sanjna compact format)
      '[2,1,3,,4]'    -> [2, 1, 3, 4]   (double-comma typo — fixed)

    Rejects (returns None):
      '[,,,]', '', nan, None             (no annotation placeholder)
      lists that are not permutations of [1,2,3,4]  (invalid rank sets)
    """
    raw = str(val).strip()
    if raw in ("", "nan", "None", "[,,,]"):
        return None

    cleaned = normalize_annotation_str(raw)
    try:
        parsed = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, list) or len(parsed) != 4:
        return None

    try:
        ranks = [int(x) for x in parsed]
    except (TypeError, ValueError):
        return None

    if sorted(ranks) != VALID_SET:
        # Not a valid permutation — e.g. [1,1,3,4] has duplicate rank
        return None

    return ranks


def consensus_ranking(rankings: list) -> list | None:
    """
    Combine N valid 4-element ranking lists into one consensus ranking.

    Algorithm:
      1 annotator  -> returned as-is (no averaging needed).
      2+ annotators -> average the rank each position received across
                       annotators, then assign final ranks 1-4 in ascending
                       order of average (lowest avg = rank 1).

    This is a Borda-count-style aggregation; ties in averaged ranks are
    broken by position index (earlier column wins), which is deterministic.

    Returns None if no valid rankings are provided.
    """
    valid = [r for r in rankings if r is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    arr   = np.array(valid, dtype=float)   # shape (n_annotators, 4)
    avg   = arr.mean(axis=0)               # shape (4,)
    order = np.argsort(avg, kind="stable") # stable: ties broken by column index
    ranks = np.empty(4, dtype=int)
    ranks[order] = np.arange(1, 5)
    return ranks.tolist()


# ============================================================
# LOAD
# ============================================================
sep = "=" * 65
print(sep)
print(" Step 0 -- JudgeLM Data Preparation")
print(sep)

if not SRC.exists():
    sys.exit(f"[ERROR] Source file not found:\n  {SRC}")

df = pd.read_csv(SRC, encoding="utf-8", low_memory=False)
N  = len(df)

print(f"\nSource : {SRC.name}")
print(f"Rows   : {N}")
print(f"Cols   : {list(df.columns)}")
print(f"Datasets: {df['dataset_name'].value_counts().to_dict()}")

errors   = []
warnings = []

# ============================================================
# STEP 1 — Audit raw annotations
# ============================================================
print(f"\n{'-'*65}")
print(" Step 1 -- Raw annotation audit")
print(f"{'-'*65}")

for col in ANNOTATOR_COLS:
    if col not in df.columns:
        errors.append(f"Annotator column missing: '{col}'")

if errors:
    for e in errors: print(f"  [ERROR] {e}")
    sys.exit("Stopping.")

print("\n  Annotator breakdown:")
for ann in ANNOTATOR_COLS:
    n_valid       = sum(1 for v in df[ann] if parse_ranking(v) is not None)
    n_placeholder = (df[ann].apply(lambda v: str(v).strip() == "[,,,]")).sum()
    n_null        = df[ann].isna().sum()

    # Count rows where normalize fixes the value
    n_typo_fixed = 0
    for v in df[ann]:
        raw = str(v).strip()
        if raw in ("", "nan", "None", "[,,,]") or pd.isna(v):
            continue
        cleaned = normalize_annotation_str(raw)
        if cleaned != raw:
            # Try parsing both; fixed must succeed where raw fails
            try:
                ast.literal_eval(raw)
                raw_ok = True
            except Exception:
                raw_ok = False
            try:
                ast.literal_eval(cleaned)
                fix_ok = True
            except Exception:
                fix_ok = False
            if not raw_ok and fix_ok:
                n_typo_fixed += 1

    print(f"  {ann:<12} valid={n_valid}  placeholder={n_placeholder}"
          f"  null={n_null}  typo_fixed={n_typo_fixed}")

# Coverage map
print("\n  Annotator row coverage:")
coverage = {}
for ann in ANNOTATOR_COLS:
    rows = df.index[df[ann].apply(lambda v: parse_ranking(v) is not None)].tolist()
    coverage[ann] = set(rows)
    if rows:
        print(f"  {ann:<12} {len(rows)} rows  [idx {min(rows)}..{max(rows)}]")
    else:
        print(f"  {ann:<12} 0 rows")

covered   = coverage["Varshini"] | coverage["Sanjna"] | coverage["Pramith"]
uncovered = set(range(N)) - covered
sp_overlap = coverage["Sanjna"] & coverage["Pramith"]
print(f"\n  Sanjna+Pramith overlap: {len(sp_overlap)} rows -> {sorted(sp_overlap)}")
print(f"  No annotation at all  : {len(uncovered)} rows -> {sorted(uncovered) or 'none'}")

# ============================================================
# STEP 2 — Parse, clean, and merge into Ranking
# ============================================================
print(f"\n{'-'*65}")
print(" Step 2 -- Parse + merge annotations -> Ranking")
print(f"{'-'*65}")

ranking_values = []
source_labels  = []
typo_fixed_rows = []

for row_i, row in df.iterrows():
    available = {}
    for ann in ANNOTATOR_COLS:
        raw  = str(row[ann]).strip()
        parsed = parse_ranking(row[ann])
        if parsed is not None:
            # Check if normalization was needed
            cleaned = normalize_annotation_str(raw)
            if cleaned != raw:
                typo_fixed_rows.append((row_i, ann, raw, cleaned))
            available[ann] = parsed

    merged = consensus_ranking(list(available.values()))
    ranking_values.append(str(merged) if merged is not None else "")
    source_labels.append("+".join(available.keys()) if available else "none")

df["Ranking"] = ranking_values

if typo_fixed_rows:
    print(f"\n  Annotation typos fixed:")
    for row_i, ann, raw, cleaned in typo_fixed_rows:
        print(f"    row {row_i} [{ann}]: {repr(raw)} -> {repr(cleaned)}")

src_counts = Counter(source_labels)
print(f"\n  Source breakdown:")
for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
    print(f"  {'':4}{src:<30} {cnt:>4} rows")

n_filled = sum(1 for r in ranking_values if r)
n_empty  = sum(1 for r in ranking_values if not r)
print(f"\n  Ranking filled : {n_filled}/{N}")
print(f"  Ranking empty  : {n_empty}  {'(rows: ' + str([i for i,l in enumerate(source_labels) if l=='none']) + ')' if n_empty else ''}")

if n_empty > 0:
    warnings.append(
        f"{n_empty} row(s) have no valid annotation from any annotator -- Ranking will be empty"
    )

# ============================================================
# STEP 3 — Fill NaN in model-response columns
# ============================================================
print(f"\n{'-'*65}")
print(" Step 3 -- Response column NaN fill")
print(f"{'-'*65}")

resp_model_cols = [
    "google/gemini-2.5-flash_response",
    "qwen/qwen3-32b_response",
    "anthropic/claude-3-haiku_response",
    "meta-llama/llama-3.2-1b-instruct_response",
]

total_filled = 0
for col in resp_model_cols:
    if col not in df.columns:
        errors.append(f"Required response column missing: '{col}'")
        print(f"  [ERROR]  {col}")
        continue
    n_null = df[col].isna().sum()
    if n_null > 0:
        null_rows = df[df[col].isna()].index.tolist()
        df[col]   = df[col].fillna("")
        total_filled += n_null
        warnings.append(
            f"'{col}': {n_null} NaN -> \"\"  (rows: {null_rows})"
        )
        print(f"  [FILL]   {col:<52} {n_null} NaN at rows {null_rows}")
    else:
        print(f"  [OK]     {col}")

if errors:
    for e in errors: print(f"\n[ERROR] {e}")
    sys.exit("Stopping.")

print(f"\n  Total cells filled: {total_filled}")
print(f"  WHY: JudgeLM inference runs on all questions; an empty response")
print(f"       will naturally score lower than real responses, which is")
print(f"       correct — those 4 rows had no model output.")

# ============================================================
# STEP 3b — Response length audit (flag outliers)
# ============================================================
print(f"\n  Response length stats (word counts):")
alias = {
    "google/gemini-2.5-flash_response":          "gemini_flash ",
    "qwen/qwen3-32b_response":                   "qwen3_32b    ",
    "anthropic/claude-3-haiku_response":          "claude_haiku ",
    "meta-llama/llama-3.2-1b-instruct_response": "llama3_2_1b  ",
}
for col in resp_model_cols:
    lengths = df[col].apply(lambda x: len(str(x).split()))
    outliers = df.index[lengths > 600].tolist()
    print(f"  {alias[col]}  min={lengths.min():<4} median={int(lengths.median()):<4}"
          f" p95={int(lengths.quantile(0.95)):<4} max={lengths.max():<5}"
          f"  outliers(>600w)={outliers or 'none'}")

print(f"\n  NOTE: row 503 llama response = 2854 words. Without truncation,")
print(f"  any pair involving llama at row 503 would exceed JudgeLM's 4096-token")
print(f"  context window. Truncation is applied in Step 1 of 1_prepare_pairs.py.")

# ============================================================
# STEP 4 — Add question_id
# ============================================================
print(f"\n{'-'*65}")
print(" Step 4 -- question_id")
print(f"{'-'*65}")

if "question_id" in df.columns:
    print("  Already present -- keeping.")
else:
    df.insert(0, "question_id", range(N))
    print(f"  Added (0-indexed, 0..{N-1}).")

# ============================================================
# STEP 5 — Final validation
# ============================================================
print(f"\n{'-'*65}")
print(" Step 5 -- Validation")
print(f"{'-'*65}")

checks = {
    "question_id present":                          "question_id" in df.columns,
    "question: no nulls":                           df["question"].isna().sum() == 0,
    "original_answer: no nulls":                    df["original_answer"].isna().sum() == 0,
    "google/gemini response: no nulls":             df["google/gemini-2.5-flash_response"].isna().sum() == 0,
    "qwen response: no nulls":                      df["qwen/qwen3-32b_response"].isna().sum() == 0,
    "claude response: no nulls":                    df["anthropic/claude-3-haiku_response"].isna().sum() == 0,
    "llama response: no nulls":                     df["meta-llama/llama-3.2-1b-instruct_response"].isna().sum() == 0,
    "Ranking column present":                       "Ranking" in df.columns,
    f"Ranking filled >= {N} rows (all covered)":    n_filled == N,
}

all_ok = True
for label, ok in checks.items():
    status = "[OK]  " if ok else "[FAIL]"
    print(f"  {status}  {label}")
    if not ok:
        all_ok = False

# Spot-check: confirm Ranking values are parseable lists
bad_ranking_rows = []
for i, val in enumerate(df["Ranking"]):
    if val == "":
        continue
    try:
        r = ast.literal_eval(str(val))
        assert isinstance(r, list) and len(r) == 4 and sorted(r) == [1,2,3,4]
    except Exception:
        bad_ranking_rows.append(i)

if bad_ranking_rows:
    print(f"  [FAIL]  Ranking values not valid permutations at rows: {bad_ranking_rows}")
    all_ok = False
else:
    print(f"  [OK]    All non-empty Ranking values are valid [1,2,3,4] permutations")

# ============================================================
# STEP 6 — Save
# ============================================================
print(f"\n{'-'*65}")
print(" Step 6 -- Save")
print(f"{'-'*65}")

if "dataset_name" not in df.columns:
    df["dataset_name"] = ""

final_cols = [c for c in OUTPUT_COL_ORDER if c in df.columns]
out_df     = df[final_cols].copy()

out_df.to_csv(DEST, index=False, encoding="utf-8")
print(f"\n  Saved  -> {DEST}")
print(f"  Cols   : {list(out_df.columns)}")
print(f"  Rows   : {len(out_df)}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{sep}")
print(" Summary")
print(sep)
print(f"  Total rows              : {N}")
print(f"  Rows used for inference : {N}  (all rows — JudgeLM does not need Ranking)")
print(f"  Rows used for metrics   : {n_filled}  (have a valid consensus Ranking)")
print(f"  Rows with empty Ranking : {n_empty}")
print(f"  Annotation typos fixed  : {len(typo_fixed_rows)}")
print(f"  Response NaN cells filled: {total_filled}")

if warnings:
    print("\n  Warnings:")
    for w in warnings:
        print(f"    [!] {w}")

if all_ok:
    print("\n  [OK] File is ready for Unity.")
    print(f"\n  Next:")
    print(f"    scp {DEST.name} <user>@unity.rc.umass.edu:~/judgelm_baseline/")
    print(f"    python 1_prepare_pairs.py")
else:
    print("\n  [FAIL] Fix the checks above before running the pipeline.")
