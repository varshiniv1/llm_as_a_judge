"""
JudgeLM inference using HuggingFace Transformers.
Works on A100 (and any GPU with cc>=7.0). No vLLM required.

Usage (called by SLURM job):
  python run_judgelm_hf.py \
      --input_dir  judgelm_baseline/data/pairs/forward \
      --output_dir judgelm_baseline/outputs/pairwise/forward \
      --model      BAAI/JudgeLM-7B-v1.0 \
      --expected_rows 529 \
      --batch_size 8
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def run_batch(model, tokenizer, prompts, max_new_tokens, device):
    """Batched greedy inference. Returns list of generated strings (new tokens only)."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3584,     # input budget: 4096 - 512 output tokens
        padding_side="left", # decoder-only needs left-padding for batched gen
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    input_len      = input_ids.shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy (temperature=0 equivalent)
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for out in out_ids:
        new_tokens = out[input_len:]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results


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
    parser.add_argument("--temperature",  type=float, default=0.0)   # kept for compat; always greedy
    parser.add_argument("--tensor_parallel", type=int, default=1)    # kept for compat; ignored
    parser.add_argument("--batch_size",   type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:    {props.name}")
        print(f"VRAM:   {props.total_memory / 1e9:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_files = sorted(in_dir.glob("*.jsonl"))
    if not pair_files:
        print(f"No JSONL files in {in_dir}", file=sys.stderr)
        sys.exit(1)

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

    # --------------------------------------------------------
    # Load model + tokenizer
    # --------------------------------------------------------
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model:     {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",          # puts model on GPU automatically
        trust_remote_code=True,
    )
    model.eval()
    print("Model ready.\n")

    # --------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------
    for pf in todo:
        out_path = out_dir / pf.name
        print(f"[RUN ] {pf.name}")
        rows = load_jsonl(pf)
        if not rows:
            print("  WARNING: empty file, skipping.")
            continue

        prompts = [
            build_prompt(
                question  = r.get("question_body", ""),
                answer1   = r.get("answer1_body",  ""),
                answer2   = r.get("answer2_body",  ""),
                reference = r.get("reference",     ""),
            )
            for r in rows
        ]

        print(f"  Prompts: {len(prompts)}  |  batch_size: {args.batch_size}")
        all_preds = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            preds = run_batch(model, tokenizer, batch, args.max_new_tokens, device)
            all_preds.extend(preds)
            done = min(i + args.batch_size, len(prompts))
            print(f"  {done}/{len(prompts)}", flush=True)

        with open(out_path, "w", encoding="utf-8") as fout:
            for row, pred_text in zip(rows, all_preds):
                result = {
                    "question_id":      row.get("question_id"),
                    "answer1_model_id": row.get("answer1_model_id"),
                    "answer2_model_id": row.get("answer2_model_id"),
                    "reverse":          row.get("reverse", False),
                    "pred_text":        pred_text,
                }
                fout.write(json.dumps(result) + "\n")

        n_written = sum(1 for _ in open(out_path) if _.strip())
        print(f"  Written: {n_written} rows  ->  {out_path}\n")

    print("All done.")


if __name__ == "__main__":
    main()
