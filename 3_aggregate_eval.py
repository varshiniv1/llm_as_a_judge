"""
Step 3 of 3: Aggregate JudgeLM pairwise outputs → rankings + full evaluation.

Reads:
  judgelm_baseline/outputs/pairwise/forward/*.jsonl  (6 files)
  judgelm_baseline/outputs/pairwise/reverse/*.jsonl  (6 files, positional bias)

Writes to  judgelm_baseline/outputs/results/ :
  JudgeLM_Ranking_vs_Human_Ranking.csv      — detailed per-question view
  JudgeLM_Final_Rankings_Comparison.csv     — compact human vs judge comparison
  positional_bias_detail.csv                — per-pair consistency
  length_bias_detail.csv                    — per-pair length vs score
  win_rate_leaderboard.csv                  — model win rates
  judgelm_baseline_summary.txt              — all metrics in one place
"""

import ast
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "human_annotated_dataset.csv"
WORK_DIR = Path("judgelm_baseline")
FWD_DIR  = WORK_DIR / "outputs" / "pairwise" / "forward"
REV_DIR  = WORK_DIR / "outputs" / "pairwise" / "reverse"
OUT_DIR  = WORK_DIR / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Alphabetical fixed order used throughout (must match prepare_pairs.py)
MODEL_ORDER = ["claude_haiku", "gemini_flash", "llama3_2_1b", "qwen3_32b"]
MODEL_IDX   = {m: i + 1 for i, m in enumerate(MODEL_ORDER)}   # 1-indexed for pred_ columns

# Response columns in the INPUT CSV (left-to-right order)
RESPONSE_COL_ORDER = ["gemini_flash", "qwen3_32b", "claude_haiku", "llama3_2_1b"]

MODEL_COLS = {
    "gemini_flash": "google/gemini-2.5-flash_response",
    "qwen3_32b":    "qwen/qwen3-32b_response",
    "claude_haiku": "anthropic/claude-3-haiku_response",
    "llama3_2_1b":  "meta-llama/llama-3.2-1b-instruct_response",
}

# Column name containing human rankings in the input CSV
HUMAN_COL = "Ranking"

PAIRS = list(itertools.combinations(MODEL_ORDER, 2))   # 6 pairs

# ============================================================
# HELPERS
# ============================================================

def parse_pred(text):
    """
    Extract (score1, score2) from JudgeLM pred_text.

    JudgeLM is trained to output two integer scores (1-10) on the first line,
    e.g. "9 7\\n[explanation...]".  Three patterns are tried in order; any
    candidate scores outside [1, 10] are rejected to prevent false matches
    from explanation text (e.g. "Response 1 has 3 examples" would naively
    produce (1, 3) without validation).

    Returns (None, None) on failure.
    """
    if not text or (isinstance(text, float) and pd.isna(text)):
        return None, None
    text = str(text).strip()

    def in_range(s1, s2):
        return 1 <= s1 <= 10 and 1 <= s2 <= 10

    # Pattern 1: explicit "Score 1: 9  Score 2: 7" anywhere in text
    m1 = re.search(r"Score\s*1\s*[:\-]\s*(\d+)", text, re.IGNORECASE)
    m2 = re.search(r"Score\s*2\s*[:\-]\s*(\d+)", text, re.IGNORECASE)
    if m1 and m2:
        s1, s2 = int(m1.group(1)), int(m2.group(1))
        if in_range(s1, s2):
            return s1, s2

    # Pattern 2: "9 7" at the very start of output (standard JudgeLM format)
    m = re.match(r"^(\d+)\s+(\d+)", text)
    if m:
        s1, s2 = int(m.group(1)), int(m.group(2))
        if in_range(s1, s2):
            return s1, s2

    # Pattern 3 (conservative fallback): only fires when the first line
    # contains NO alphabetic characters, meaning it is a pure score line
    # (e.g. "9, 7" or "9/7") that Pattern 2 missed due to non-space separators.
    # This guards against false matches from explanation sentences like
    # "Response 1 has 3 examples" which would naively yield (1, 3).
    first_line = text.split("\n")[0].strip()
    if not re.search(r"[a-zA-Z]", first_line):
        nums = re.findall(r"\d+", first_line)
        if len(nums) >= 2:
            s1, s2 = int(nums[0]), int(nums[1])
            if in_range(s1, s2):
                return s1, s2

    return None, None


def score_to_winner(s1, s2):
    """Returns 'A', 'B', 'tie', or None (parse failure)."""
    if s1 is None or s2 is None:
        return None
    if s1 > s2:
        return "A"
    if s2 > s1:
        return "B"
    return "tie"


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def safe_parse_list(val):
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return None


def wins_to_ranking(wins_dict):
    """
    Convert {model: win_score} to {model: rank}.
    Rank 1 = best. Ties share the same rank.
    """
    sorted_models = sorted(wins_dict, key=lambda m: wins_dict[m], reverse=True)
    rank_map = {}
    current_rank = 1
    for i, model in enumerate(sorted_models):
        if i > 0 and wins_dict[model] < wins_dict[sorted_models[i - 1]]:
            current_rank = i + 1
        rank_map[model] = current_rank
    return rank_map


# ============================================================
# LOAD INPUT DATA
# ============================================================
print("=" * 60)
print(" JudgeLM Baseline Evaluation")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
N  = len(df)
print(f"\nLoaded {N} rows from {CSV_PATH}")

# Parse human rankings
if HUMAN_COL not in df.columns:
    raise ValueError(
        f"Human ranking column '{HUMAN_COL}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )
df["_human_ranking"] = df[HUMAN_COL].apply(safe_parse_list)

valid_human = df["_human_ranking"].notna().sum()
print(f"Valid human rankings: {valid_human}/{N}")

# ============================================================
# PARSE FORWARD PAIRWISE OUTPUTS
# ============================================================
print("\n--- Parsing forward judgments ---")

# fwd_data[q_id][(model_a, model_b)] = {s1, s2, winner, pred_text}
fwd_data = {i: {} for i in range(N)}
fwd_parse_stats = {}   # pair_name -> {ok, fail, missing}

for model_a, model_b in PAIRS:
    pair_name = f"{model_a}__vs__{model_b}"
    path      = FWD_DIR / f"{pair_name}.jsonl"
    rows      = load_jsonl(path)

    ok = fail = 0
    for row in rows:
        q_id = row.get("question_id")
        if q_id is None or q_id >= N:
            continue
        s1, s2   = parse_pred(row.get("pred_text", ""))
        winner   = score_to_winner(s1, s2)
        fwd_data[q_id][(model_a, model_b)] = {
            "s1": s1, "s2": s2, "winner": winner,
            "pred_text": row.get("pred_text", ""),
        }
        if winner is not None:
            ok += 1
        else:
            fail += 1

    missing = N - len(rows)
    status  = "OK" if (ok + fail + missing) >= N else "INCOMPLETE"
    fwd_parse_stats[pair_name] = {"ok": ok, "fail": fail, "missing": missing}
    print(f"  {pair_name}: {ok} parsed | {fail} failed | {missing} missing  [{status}]")

total_fwd_ok   = sum(s["ok"]      for s in fwd_parse_stats.values())
total_fwd_fail = sum(s["fail"]    for s in fwd_parse_stats.values())
total_fwd_miss = sum(s["missing"] for s in fwd_parse_stats.values())
total_fwd      = total_fwd_ok + total_fwd_fail + total_fwd_miss
fwd_parse_rate = total_fwd_ok / total_fwd if total_fwd else 0
print(f"\n  TOTAL forward: {total_fwd_ok}/{total_fwd} parsed  ({fwd_parse_rate*100:.1f}%)"
      f"  |  {total_fwd_fail} failed  |  {total_fwd_miss} missing")

# ============================================================
# PARSE REVERSE PAIRWISE OUTPUTS (positional bias)
# ============================================================
print("\n--- Parsing reverse judgments (positional bias) ---")

# rev_data[q_id][(model_a, model_b)] = winner *normalized* to (model_a, model_b) frame
rev_data = {i: {} for i in range(N)}

for model_a, model_b in PAIRS:
    pair_name = f"{model_a}__vs__{model_b}"
    path      = REV_DIR / f"{pair_name}.jsonl"
    rows      = load_jsonl(path)

    ok = fail = 0
    for row in rows:
        q_id = row.get("question_id")
        if q_id is None or q_id >= N:
            continue
        s1, s2     = parse_pred(row.get("pred_text", ""))
        winner_raw = score_to_winner(s1, s2)   # in reversed frame (model_b is A, model_a is B)

        # Normalize back to (model_a, model_b) frame
        if winner_raw == "A":
            winner_norm = "B"     # model_b was presented as A → model_b wins
        elif winner_raw == "B":
            winner_norm = "A"     # model_a was presented as B → model_a wins
        else:
            winner_norm = winner_raw   # tie or None

        rev_data[q_id][(model_a, model_b)] = {
            "s1_raw": s1, "s2_raw": s2,
            "winner_raw": winner_raw,
            "winner_norm": winner_norm,
        }
        if winner_raw is not None:
            ok += 1
        else:
            fail += 1

    print(f"  {pair_name}: {ok} parsed | {fail} failed")

# ============================================================
# AGGREGATE TO 4-WAY RANKINGS
# ============================================================
print("\n--- Aggregating to 4-way rankings ---")

ranking_rows = []

for q_id in range(N):
    row   = df.iloc[q_id]
    pairs = fwd_data[q_id]

    wins       = {m: 0.0 for m in MODEL_ORDER}
    valid_ct   = 0
    failed_ct  = 0
    pred_map   = {}    # column_name → "A" / "B" / "tie" / ""

    for model_a, model_b in PAIRS:
        i, j     = MODEL_IDX[model_a], MODEL_IDX[model_b]
        col_name = f"pred_{i}_vs_{j}"

        entry  = pairs.get((model_a, model_b))
        if entry is None:
            failed_ct += 1
            pred_map[col_name] = ""
            continue

        winner = entry["winner"]
        if winner is None:
            failed_ct += 1
            pred_map[col_name] = ""
        elif winner == "A":
            wins[model_a] += 1.0
            valid_ct += 1
            pred_map[col_name] = "A"
        elif winner == "B":
            wins[model_b] += 1.0
            valid_ct += 1
            pred_map[col_name] = "B"
        elif winner == "tie":
            wins[model_a] += 0.5
            wins[model_b] += 0.5
            valid_ct += 1
            pred_map[col_name] = "tie"

    rank_map   = wins_to_ranking(wins)
    best_model = min(rank_map, key=rank_map.get)

    ranking_rows.append({
        "question_id": q_id,
        "question":    row["question"],
        "reference":   row["original_answer"],
        "model_order": str(MODEL_ORDER),
        "judge_ranking":   str([rank_map[m] for m in MODEL_ORDER]),
        "judge_best_model": best_model,
        "wins":            str([wins[m] for m in MODEL_ORDER]),
        "valid_pairwise_judgments":  valid_ct,
        "failed_pairwise_judgments": failed_ct,
        "synthetic_used": False,
        **pred_map,
    })

# Build DataFrame with columns in correct order
pred_cols = []
for model_a, model_b in PAIRS:
    pred_cols.append(f"pred_{MODEL_IDX[model_a]}_vs_{MODEL_IDX[model_b]}")

col_order = [
    "question_id", "question", "reference", "model_order",
    "judge_ranking", "judge_best_model", "wins",
    "valid_pairwise_judgments", "failed_pairwise_judgments", "synthetic_used",
] + pred_cols

ranking_df = pd.DataFrame(ranking_rows)
for c in pred_cols:
    if c not in ranking_df.columns:
        ranking_df[c] = ""
ranking_df = ranking_df[col_order]

out_detailed = OUT_DIR / "JudgeLM_Ranking_vs_Human_Ranking.csv"
ranking_df.to_csv(out_detailed, index=False)
print(f"  Saved: {out_detailed}")

# ============================================================
# BUILD FINAL COMPARISON CSV
# ============================================================
print("\n--- Building final comparison CSV ---")

final_rows = []
for q_id in range(N):
    row = df.iloc[q_id]
    rd  = ranking_rows[q_id]

    j_rank_in_model_order = safe_parse_list(rd["judge_ranking"])
    rank_map = dict(zip(MODEL_ORDER, j_rank_in_model_order))
    # Remap to the RESPONSE_COL_ORDER used in the comparison CSV
    judgelm_ranking = [rank_map[m] for m in RESPONSE_COL_ORDER]

    final_rows.append({
        "question":        row["question"],
        "original_answer": row["original_answer"],
        "dataset_name":    row.get("dataset_name", ""),
        "google/gemini-2.5-flash_response":          row[MODEL_COLS["gemini_flash"]],
        "qwen/qwen3-32b_response":                   row[MODEL_COLS["qwen3_32b"]],
        "anthropic/claude-3-haiku_response":          row[MODEL_COLS["claude_haiku"]],
        "meta-llama/llama-3.2-1b-instruct_response": row[MODEL_COLS["llama3_2_1b"]],
        "Ranking":          str(row.get(HUMAN_COL, row.get("Ranking", ""))),
        "JudgeLM_Ranking":  str(judgelm_ranking),
    })

final_df = pd.DataFrame(final_rows)
out_final = OUT_DIR / "JudgeLM_Final_Rankings_Comparison.csv"
final_df.to_csv(out_final, index=False)
print(f"  Saved: {out_final}")

# ============================================================
# PAIRWISE ACCURACY
# — fraction of (question, pair) decisions where JudgeLM's
#   preferred model matches the human-annotated ranking.
#   Ties in JudgeLM output are excluded (no preference expressed).
#   This is the primary per-judgment quality metric for the paper.
# ============================================================
print("\n--- Computing pairwise accuracy ---")

# Build a quick lookup: model -> its index in RESPONSE_COL_ORDER
resp_col_idx = {m: i for i, m in enumerate(RESPONSE_COL_ORDER)}

pa_agree = pa_total = 0
for q_id in range(N):
    h_rank_list = df.iloc[q_id]["_human_ranking"]
    if h_rank_list is None:
        continue

    for model_a, model_b in PAIRS:
        entry = fwd_data[q_id].get((model_a, model_b))
        if entry is None or entry["winner"] is None or entry["winner"] == "tie":
            continue   # skip parse failures and ties

        judge_winner = entry["winner"]   # "A" = model_a preferred, "B" = model_b

        # Human preference derived from ranking
        rank_a = h_rank_list[resp_col_idx[model_a]]
        rank_b = h_rank_list[resp_col_idx[model_b]]
        if rank_a == rank_b:
            continue   # genuine tie in human ranking (shouldn't happen with strict ranks)
        human_winner = "A" if rank_a < rank_b else "B"   # lower rank = better

        pa_total  += 1
        if judge_winner == human_winner:
            pa_agree += 1

pairwise_accuracy = pa_agree / pa_total if pa_total else 0
print(f"  Pairwise judgments (non-tie): {pa_total}")
print(f"  Agree with human:             {pa_agree}  ({pairwise_accuracy*100:.1f}%)")
print(f"  (random baseline = 50.0%)")

# ============================================================
# HUMAN ALIGNMENT METRICS  (full set + quality subset)
# quality subset = questions where all 6 pairwise judgments parsed OK
# ============================================================
print("\n--- Computing human alignment metrics ---")

# Which questions have all 6 pairs successfully judged?
all_pairs_ok = set()
for q_id in range(N):
    if all(
        fwd_data[q_id].get((a, b), {}).get("winner") is not None
        for a, b in PAIRS
    ):
        all_pairs_ok.add(q_id)
print(f"  Questions with all 6 pairs parsed: {len(all_pairs_ok)}/{N}")


def _alignment_metrics(q_ids, label):
    spearmans, kendalls = [], []
    top1_hits = exact_hits = valid_n = 0
    y_true_best, y_pred_best = [], []

    for q_id in q_ids:
        h_rank = df.iloc[q_id]["_human_ranking"]
        rd     = ranking_rows[q_id]

        if h_rank is None:
            continue

        j_rank_model_order = safe_parse_list(rd["judge_ranking"])
        if j_rank_model_order is None:
            continue

        # Remap judge ranking from MODEL_ORDER -> RESPONSE_COL_ORDER
        rmap   = dict(zip(MODEL_ORDER, j_rank_model_order))
        j_rank = [rmap[m] for m in RESPONSE_COL_ORDER]

        valid_n += 1
        rho, _ = spearmanr(h_rank, j_rank)
        tau, _ = kendalltau(h_rank, j_rank)
        spearmans.append(rho)
        kendalls.append(tau)

        h_best = h_rank.index(min(h_rank))
        j_best = j_rank.index(min(j_rank))
        y_true_best.append(h_best)
        y_pred_best.append(j_best)

        if h_best == j_best:
            top1_hits += 1
        if list(h_rank) == list(j_rank):
            exact_hits += 1

    if valid_n == 0:
        return {}

    result = {
        "label":            label,
        "n":                valid_n,
        "spearman_mean":    np.nanmean(spearmans),
        "spearman_median":  np.nanmedian(spearmans),
        "spearman_std":     np.nanstd(spearmans),
        "kendall_mean":     np.nanmean(kendalls),
        "kendall_median":   np.nanmedian(kendalls),
        "kendall_std":      np.nanstd(kendalls),
        "top1_accuracy":    top1_hits / valid_n,
        "exact_rank_match": exact_hits / valid_n,
        "top1_count":       top1_hits,
        "exact_count":      exact_hits,
    }
    if y_true_best:
        try:
            result["cohens_kappa_top1"] = cohen_kappa_score(y_true_best, y_pred_best)
        except Exception:
            result["cohens_kappa_top1"] = float("nan")
    return result


align      = _alignment_metrics(range(N),      "all questions")
align_qs   = _alignment_metrics(sorted(all_pairs_ok), "quality subset (all 6 pairs OK)")
valid_n    = align.get("n", 0)
top1_hits  = align.get("top1_count", 0)
exact_hits = align.get("exact_count", 0)

for a in [align, align_qs]:
    if not a:
        continue
    lbl = a["label"]
    print(f"\n  [{lbl}]  n={a['n']}")
    print(f"  Spearman rho  mean={a['spearman_mean']:.4f}  median={a['spearman_median']:.4f}")
    print(f"  Kendall tau   mean={a['kendall_mean']:.4f}  median={a['kendall_median']:.4f}")
    print(f"  Top-1 Acc     {a['top1_accuracy']:.4f}  ({a['top1_count']}/{a['n']})")
    print(f"  Exact Match   {a['exact_rank_match']:.4f}  ({a['exact_count']}/{a['n']})")
    if "cohens_kappa_top1" in a:
        print(f"  Cohen kappa   {a['cohens_kappa_top1']:.4f}")

# ============================================================
# WIN RATE LEADERBOARD
# ============================================================
print("\n--- Win rate leaderboard ---")

win_stats = {m: {"wins": 0, "total": 0} for m in MODEL_ORDER}
for q_id in range(N):
    for model_a, model_b in PAIRS:
        entry = fwd_data[q_id].get((model_a, model_b))
        if entry is None:
            continue
        winner = entry["winner"]
        for m in [model_a, model_b]:
            win_stats[m]["total"] += 1
        if winner == "A":
            win_stats[model_a]["wins"] += 1
        elif winner == "B":
            win_stats[model_b]["wins"] += 1
        elif winner == "tie":
            win_stats[model_a]["wins"] += 0.5
            win_stats[model_b]["wins"] += 0.5

lb_rows = []
for m, s in win_stats.items():
    wr = s["wins"] / s["total"] if s["total"] > 0 else 0
    lb_rows.append({"model": m, "wins": s["wins"], "total": s["total"], "win_rate": wr})
    print(f"  {m}: {s['wins']:.1f}/{s['total']}  ({wr*100:.1f}%)")

leaderboard_df = pd.DataFrame(lb_rows).sort_values("win_rate", ascending=False).reset_index(drop=True)
out_lb = OUT_DIR / "win_rate_leaderboard.csv"
leaderboard_df.to_csv(out_lb, index=False)
print(f"  Saved: {out_lb}")

# ============================================================
# POSITIONAL BIAS ANALYSIS
# ============================================================
print("\n--- Positional bias analysis ---")

pos_rows = []
consistent = pos1_fwd = pos1_rev = total_pos = 0

for q_id in range(N):
    for model_a, model_b in PAIRS:
        fwd = fwd_data[q_id].get((model_a, model_b))
        rev = rev_data[q_id].get((model_a, model_b))

        if fwd is None or rev is None:
            continue
        fw, rn = fwd["winner"], rev["winner_norm"]
        if fw is None or rn is None:
            continue

        total_pos += 1
        is_consistent = (fw == rn)
        if is_consistent:
            consistent += 1
        if fwd["winner"] == "A":
            pos1_fwd += 1
        if rev["winner_raw"] == "A":   # raw "A" in reverse = position-1 preference
            pos1_rev += 1

        pos_rows.append({
            "question_id":        q_id,
            "model_a":            model_a,
            "model_b":            model_b,
            "fwd_winner":         fw,
            "rev_winner_norm":    rn,
            "consistent":         is_consistent,
            "fwd_score1":         fwd["s1"],
            "fwd_score2":         fwd["s2"],
            "rev_score1_raw":     rev["s1_raw"],
            "rev_score2_raw":     rev["s2_raw"],
        })

pos_df  = pd.DataFrame(pos_rows)
out_pos = OUT_DIR / "positional_bias_detail.csv"
pos_df.to_csv(out_pos, index=False)

consistency_rate = consistent / total_pos if total_pos else 0
p1_pref_fwd      = pos1_fwd   / total_pos if total_pos else 0
p1_pref_rev      = pos1_rev   / total_pos if total_pos else 0

print(f"  Pair evaluations:         {total_pos}")
print(f"  Consistency rate:         {consistency_rate:.4f}")
print(f"  Position-1 pref (fwd):    {p1_pref_fwd:.4f}")
print(f"  Position-1 pref (rev):    {p1_pref_rev:.4f}")
print(f"  (0.5 = unbiased; >0.6 = notable position-1 bias)")
print(f"  Saved: {out_pos}")

pos_bias = {
    "total_pair_evaluations": total_pos,
    "consistency_rate":       consistency_rate,
    "pos1_preference_fwd":    p1_pref_fwd,
    "pos1_preference_rev":    p1_pref_rev,
    "pos1_bias_delta":        abs(p1_pref_fwd - 0.5),
    "inconsistency_rate":     1 - consistency_rate,
}

# ============================================================
# LENGTH BIAS ANALYSIS
# ============================================================
print("\n--- Length bias analysis ---")

len_rows = []
longer_wins = longer_total = 0

for q_id in range(N):
    row = df.iloc[q_id]
    for model_a, model_b in PAIRS:
        entry = fwd_data[q_id].get((model_a, model_b))
        if entry is None or entry["winner"] is None:
            continue

        len_a = len(str(row[MODEL_COLS[model_a]]).split())
        len_b = len(str(row[MODEL_COLS[model_b]]).split())
        winner = entry["winner"]
        s1, s2 = entry["s1"], entry["s2"]

        if len_a == len_b:
            longer_model = "tie"
        else:
            longer_model = "A" if len_a > len_b else "B"

        if longer_model != "tie" and winner != "tie":
            longer_total += 1
            if longer_model == winner:
                longer_wins += 1

        len_rows.append({
            "question_id":  q_id,
            "model_a":      model_a,
            "model_b":      model_b,
            "len_a_words":  len_a,
            "len_b_words":  len_b,
            "len_diff":     len_a - len_b,   # positive → A is longer
            "score1":       s1,
            "score2":       s2,
            "score_diff":   (s1 or 0) - (s2 or 0),   # positive → A scored higher
            "winner":       winner,
            "longer_model": longer_model,
            "longer_wins":  (longer_model == winner) if (longer_model != "tie" and winner != "tie") else None,
        })

len_df  = pd.DataFrame(len_rows)
out_len = OUT_DIR / "length_bias_detail.csv"
len_df.to_csv(out_len, index=False)

longer_wins_rate = longer_wins / longer_total if longer_total else 0

# Correlation: len_diff vs score_diff
valid_len = len_df.dropna(subset=["len_diff", "score_diff"])
if len(valid_len) > 10:
    pearson_r,  pearson_p  = pearsonr(valid_len["len_diff"],  valid_len["score_diff"])
    spearman_r, spearman_p = spearmanr(valid_len["len_diff"], valid_len["score_diff"])
else:
    pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

# Per-model: are longer responses systematically preferred?
model_length_pref = {}
for m in MODEL_ORDER:
    sub_a = len_df[len_df["model_a"] == m]
    sub_b = len_df[len_df["model_b"] == m]
    wins_as_a = ((sub_a["len_a_words"] > sub_a["len_b_words"]) & (sub_a["winner"] == "A")).sum()
    wins_as_b = ((sub_b["len_b_words"] > sub_b["len_a_words"]) & (sub_b["winner"] == "B")).sum()
    total_longer = (
        ((sub_a["len_a_words"] > sub_a["len_b_words"]) & sub_a["winner"].isin(["A", "B"])).sum() +
        ((sub_b["len_b_words"] > sub_b["len_a_words"]) & sub_b["winner"].isin(["A", "B"])).sum()
    )
    model_length_pref[m] = (wins_as_a + wins_as_b) / total_longer if total_longer > 0 else float("nan")

print(f"  Longer answer wins:       {longer_wins}/{longer_total}  ({longer_wins_rate*100:.1f}%)")
print(f"  Pearson r (len vs score): {pearson_r:.4f}  (p={pearson_p:.4f})")
print(f"  Spearman r:               {spearman_r:.4f}  (p={spearman_p:.4f})")
print(f"  (r ≈ 0 = no bias; r > 0.3 = notable length bias)")
print(f"  Saved: {out_len}")

len_bias = {
    "total_non_tie_pairs":       longer_total,
    "longer_wins_rate":          longer_wins_rate,
    "pearson_r_len_score":       pearson_r,
    "pearson_p":                 pearson_p,
    "spearman_r_len_score":      spearman_r,
    "spearman_p":                spearman_p,
    "per_model_longer_wins_rate": model_length_pref,
}

# ============================================================
# WRITE SUMMARY REPORT
# ============================================================
print("\n--- Writing summary report ---")

def bias_label(val, neutral=0.5, hi=0.6, mid=0.55):
    if val > hi:   return "HIGH"
    if val > mid:  return "MODERATE"
    return "LOW"

report_path = OUT_DIR / "judgelm_baseline_summary.txt"
with open(report_path, "w", encoding="utf-8") as f:

    f.write("=" * 65 + "\n")
    f.write("  JudgeLM Baseline Evaluation -- Summary Report\n")
    f.write("=" * 65 + "\n\n")

    f.write("[Dataset]\n")
    f.write(f"  Input CSV:       {CSV_PATH}\n")
    f.write(f"  N questions:     {N}\n")
    f.write(f"  Human col:       {HUMAN_COL}\n")
    f.write(f"  Models:          {MODEL_ORDER}\n\n")

    f.write("[Parse Success Rate]\n")
    f.write(f"  Forward judgments parsed:  {total_fwd_ok}/{total_fwd}  ({fwd_parse_rate*100:.1f}%)\n")
    f.write(f"  Failed:                    {total_fwd_fail}\n")
    f.write(f"  Missing (file not found):  {total_fwd_miss}\n")
    f.write(f"  Per-file breakdown:\n")
    for pname, ps in fwd_parse_stats.items():
        tot = ps['ok'] + ps['fail'] + ps['missing']
        rate = ps['ok'] / tot * 100 if tot else 0
        f.write(f"    {pname:<42}  {ps['ok']}/{tot}  ({rate:.0f}%)\n")
    f.write(f"  Questions with all 6 pairs OK: {len(all_pairs_ok)}/{N}\n\n")

    f.write("[Pairwise Accuracy]\n")
    f.write(f"  (fraction of per-pair JudgeLM decisions that match human preference)\n")
    f.write(f"  (tie judgments excluded; random baseline = 50.0%)\n")
    f.write(f"  Non-tie pairwise judgments: {pa_total}\n")
    f.write(f"  Agreed with human:          {pa_agree}  ({pairwise_accuracy*100:.1f}%)\n\n")

    def _write_align(f, a, indent="  "):
        if not a:
            f.write(f"{indent}(no data)\n")
            return
        f.write(f"{indent}N evaluated:           {a['n']}\n")
        f.write(f"{indent}Spearman rho  mean:    {a['spearman_mean']:.4f}\n")
        f.write(f"{indent}Spearman rho  median:  {a['spearman_median']:.4f}\n")
        f.write(f"{indent}Spearman rho  std:     {a['spearman_std']:.4f}\n")
        f.write(f"{indent}Kendall tau   mean:    {a['kendall_mean']:.4f}\n")
        f.write(f"{indent}Kendall tau   median:  {a['kendall_median']:.4f}\n")
        f.write(f"{indent}Kendall tau   std:     {a['kendall_std']:.4f}\n")
        f.write(f"{indent}Top-1 Accuracy:        {a['top1_accuracy']:.4f}"
                f"  ({a['top1_count']}/{a['n']})\n")
        f.write(f"{indent}Exact Rank Match:      {a['exact_rank_match']:.4f}"
                f"  ({a['exact_count']}/{a['n']})\n")
        if "cohens_kappa_top1" in a:
            f.write(f"{indent}Cohen kappa (top-1):   {a['cohens_kappa_top1']:.4f}\n")

    f.write("[Human Alignment -- All Questions]\n")
    _write_align(f, align)
    f.write("\n")

    f.write("[Human Alignment -- Quality Subset (all 6 pairs parsed)]\n")
    _write_align(f, align_qs)
    f.write("\n")

    f.write("[Win Rate Leaderboard]\n")
    for _, lb_r in leaderboard_df.iterrows():
        f.write(f"  {lb_r['model']:<20}  {lb_r['wins']:.1f} wins / {lb_r['total']}"
                f"  ({lb_r['win_rate']*100:.1f}%)\n")
    f.write("\n")

    f.write("[Positional Bias]\n")
    f.write(f"  Pair evaluations:      {pos_bias['total_pair_evaluations']}\n")
    f.write(f"  Consistency rate:      {pos_bias['consistency_rate']:.4f}\n")
    f.write(f"    (same winner regardless of presentation order)\n")
    f.write(f"  Inconsistency rate:    {pos_bias['inconsistency_rate']:.4f}\n")
    f.write(f"  Position-1 pref (fwd): {pos_bias['pos1_preference_fwd']:.4f}\n")
    f.write(f"  Position-1 pref (rev): {pos_bias['pos1_preference_rev']:.4f}\n")
    f.write(f"  Pos-1 bias delta:      {pos_bias['pos1_bias_delta']:.4f}\n")
    f.write(f"  Assessment:            {bias_label(pos_bias['pos1_preference_fwd'])}"
            f" positional bias\n\n")

    f.write("[Length Bias]\n")
    f.write(f"  Non-tie pair evals:    {len_bias['total_non_tie_pairs']}\n")
    f.write(f"  Longer answer wins:    {len_bias['longer_wins_rate']:.4f}\n")
    f.write(f"    (0.5 = no bias; >0.6 = notable)\n")
    f.write(f"  Pearson r (len, score):{len_bias['pearson_r_len_score']:.4f}"
            f"  p={len_bias['pearson_p']:.4f}\n")
    f.write(f"  Spearman r:            {len_bias['spearman_r_len_score']:.4f}"
            f"  p={len_bias['spearman_p']:.4f}\n")
    f.write(f"  Per-model longer-wins rate:\n")
    for m, r in len_bias["per_model_longer_wins_rate"].items():
        f.write(f"    {m:<20}  {r:.4f}\n")
    f.write(f"  Assessment:            {bias_label(len_bias['longer_wins_rate'])}"
            f" length bias\n\n")

    f.write("[Output Files]\n")
    for fname in [
        "JudgeLM_Ranking_vs_Human_Ranking.csv",
        "JudgeLM_Final_Rankings_Comparison.csv",
        "win_rate_leaderboard.csv",
        "positional_bias_detail.csv",
        "length_bias_detail.csv",
        "judgelm_baseline_summary.txt",
    ]:
        f.write(f"  {fname}\n")

print(f"  Saved: {report_path}")
print(f"\nAll outputs in: {OUT_DIR}")
print("\nDone.")
