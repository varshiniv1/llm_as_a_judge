"""
JudgeLM inference via vLLM (fast batched GPU inference).

Usage (called by SLURM job — do not run directly):
  python run_judgelm_vllm.py \
      --input_dir  judgelm_baseline/data/pairs/forward \
      --output_dir judgelm_baseline/outputs/pairwise/forward \
      --model      BAAI/JudgeLM-7B-v1.0 \
      --expected_rows 529

Processes every *.jsonl in input_dir. Skips files already complete.
Outputs one *.jsonl per input file with a pred_text field added.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# JudgeLM-7B config.json reports max_position_embeddings=2048 (LLaMA-1
# base), but it was fine-tuned for 4096. Newer vLLM rejects max_model_len
# > config value unless this flag is set. Must be set before vllm import.
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

from vllm import LLM, SamplingParams

# ============================================================
# JudgeLM prompt (with reference answer)
# ============================================================
SYSTEM_MSG = (
    "We would like to request your feedback on the performance of two AI assistants "
    "in response to the user question displayed above. Please rate the helpfulness, "
    "relevance, accuracy, level of details of their responses. Each assistant receives "
    "an overall score on a scale of 1 to 10, where a higher score indicates better "
    "overall performance.\n"
    "Please first output a single line containing only two values indicating the scores "
    "for Assistant 1 and Assistant 2, respectively. The two scores are separated by a "
    "space. In the subsequent line, please provide a comprehensive explanation of your "
    "evaluation, avoiding any potential bias and ensuring that the order in which the "
    "responses were presented does not affect your judgment."
)


def build_prompt(question: str, answer1: str, answer2: str, reference: str = "") -> str:
    parts = ["[Question]", question.strip()]
    if reference and reference.strip():
        parts += [
            "[The Start of Reference Answer]",
            reference.strip(),
            "[The End of Reference Answer]",
        ]
    parts += [
        "[The Start of Assistant 1's Answer]",
        answer1.strip(),
        "[The End of Assistant 1's Answer]",
        "[The Start of Assistant 2's Answer]",
        answer2.strip(),
        "[The End of Assistant 2's Answer]",
        "[System]",
        SYSTEM_MSG,
    ]
    return "\n".join(parts)


# ============================================================
# Helpers
# ============================================================
def load_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def already_complete(out_path: Path, expected: int) -> bool:
    if not out_path.exists():
        return False
    n = sum(1 for _ in open(out_path, encoding="utf-8") if _.strip())
    return n >= expected


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",    required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--model",        default="BAAI/JudgeLM-7B-v1.0")
    parser.add_argument("--expected_rows", type=int, default=529)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature",  type=float, default=0.0)   # greedy
    parser.add_argument("--tensor_parallel", type=int, default=1)    # GPUs
    args = parser.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_files = sorted(in_dir.glob("*.jsonl"))
    if not pair_files:
        print(f"No JSONL files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Find which files still need running
    todo = []
    for pf in pair_files:
        out_path = out_dir / pf.name
        if already_complete(out_path, args.expected_rows):
            print(f"[SKIP] {pf.name}  (already {args.expected_rows} rows)")
        else:
            todo.append(pf)

    if not todo:
        print("All pairs already complete. Nothing to do.")
        return

    print(f"\nLoading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,      # increase if you hit OOM / truncation warnings
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=None,
    )

    for pf in todo:
        out_path = out_dir / pf.name
        print(f"\n[RUN ] {pf.name}")

        rows = load_jsonl(pf)
        if not rows:
            print(f"  WARNING: empty file, skipping.")
            continue

        # Build all prompts for this pair file at once
        prompts = [
            build_prompt(
                question  = r.get("question_body", ""),
                answer1   = r.get("answer1_body",  ""),
                answer2   = r.get("answer2_body",  ""),
                reference = r.get("reference",     ""),
            )
            for r in rows
        ]

        print(f"  Prompts: {len(prompts)}")
        outputs = llm.generate(prompts, sampling_params)

        with open(out_path, "w", encoding="utf-8") as fout:
            for row, out in zip(rows, outputs):
                pred_text = out.outputs[0].text.strip()
                result = {
                    "question_id":    row.get("question_id"),
                    "answer1_model_id": row.get("answer1_model_id"),
                    "answer2_model_id": row.get("answer2_model_id"),
                    "reverse":        row.get("reverse", False),
                    "pred_text":      pred_text,
                }
                fout.write(json.dumps(result) + "\n")

        n_written = sum(1 for _ in open(out_path) if _.strip())
        print(f"  Written: {n_written} rows  →  {out_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
