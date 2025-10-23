import argparse
import pandas as pd
from itertools import product
import numpy as np

SUMM_SHORT_TMPL = (
    'Context: "{story}"\n'
    'Task: "Write a concise summary of the context of about {target_words} words. '
    'Output only the summary."'
)

SUMM_LONG_TMPL = (
    'Context: "{story}"\n'
    'Task: "Write a concise summary of the context of about {target_words} words. '
    'Output only the summary."'
)

QA_TMPL = (
    'Context: "{story}"\n'
    'Question: "{question}"\n'
    'Task: "Answer the question using only the information in the context. '
    'Output only the answer phrase."'
)

def sanitize(text: str) -> str:
    """Ensure strings are safe inside double quotes."""
    if text is None:
        return ""
    return str(text).replace('"', '\\"')

def build_prompt(row):
    task = str(row["task"]).strip().upper()
    story = sanitize(row.get("story", ""))

    if task == "QA":
        question = sanitize(row.get("question", ""))
        return QA_TMPL.format(story=story, question=question)

    elif task in ("SUMM_SHORT", "SUMM_LONG"):
        # target_words present in your CSV; fallback if missing
        tw = row.get("target_words", "")
        try:
            tw = int(float(tw))
        except Exception:
            tw = 120 if task == "SUMM_SHORT" else 200
        tmpl = SUMM_SHORT_TMPL if task == "SUMM_SHORT" else SUMM_LONG_TMPL
        return tmpl.format(story=story, target_words=tw)

    else:
        raise ValueError(f"Unknown task type: {task}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV with 150 items")
    ap.add_argument("--out_csv", default="run_plan.csv", help="Output run-plan CSV")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["LLaMA-2-7B", "Mistral-7B", "Falcon-7B"],
        help="Model names",
    )
    ap.add_argument(
        "--quant_levels",
        nargs="+",
        default=["FP16", "INT8", "INT4"],
        help="Quantization levels",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = ap.parse_args()

    # Load CSV with your exact columns
    df = pd.read_csv(args.in_csv)

    required_cols = {
        "row_id", "story", "question", "answer", "summary",
        "story_tokens", "answer_tokens", "summ_bucket",
        "story_bucket", "task", "target_words"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    # Build prompt column
    df["prompt"] = df.apply(build_prompt, axis=1)

    # Minimal subset to expand
    base = df[["row_id", "task", "prompt"]].copy()
    base["task"] = base["task"].str.upper().str.strip()

    # Cartesian product over models × quantization levels
    combos = list(product(args.models, args.quant_levels))
    records = []
    for _, row in base.iterrows():
        for model, q in combos:
            records.append({
                "row_id": int(row["row_id"]),
                "task_type": row["task"],
                "prompt": row["prompt"],
                "model": model,
                "quantization_level": q,
                "run_status": "PENDING",
            })

    run_df = pd.DataFrame.from_records(records)

    # RANDOMIZE order to avoid consecutive runs of the same model/task
    run_df = run_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Assign run_id AFTER shuffling
    run_df.insert(0, "run_id", [f"run_{i:04d}" for i in range(1, len(run_df) + 1)])

    # Final column order
    run_df = run_df[
        [
            "run_id",
            "run_status",
            "model",
            "quantization_level",
            "task_type",
            "row_id",
            "prompt",
        ]
    ]

    run_df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} with {len(run_df)} randomized runs.")

if __name__ == "__main__":
    main()
