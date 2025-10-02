# plan_subset.py
# Usage example:
#   python plan_subset.py --qa_n 50 --summ_short_n 50 --summ_long_n 50 --seed 42 \
#     --dataset glnmario/news-qa-summarization --split train --outdir ./subset_plan
#
# Outputs:
#   ./subset_plan/qa_selected.csv
#   ./subset_plan/summ_short_selected.csv
#   ./subset_plan/summ_long_selected.csv
#   ./subset_plan/manifest_all_tasks.csv
#   ./subset_plan/length_targets.txt

import argparse
import os
from pathlib import Path
import random
import math

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------- helpers ---------------------- #

def to_text(x):
    """Return a plain string from possibly nested/list/dict structures."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # try common keys first
        for k in ("text", "answer", "answers", "summary", "label", "value"):
            if k in x:
                return to_text(x[k])
        return " ".join(to_text(v) for v in x.values())
    if isinstance(x, (list, tuple, set)):
        parts = [to_text(e) for e in x]
        parts = [p for p in parts if p]
        return parts[0] if parts else ""
    return str(x)

def count_tokens(text, tok):
    txt = to_text(text)
    if not txt:
        return 0
    # model-agnostic count; GPT-2 BPE by default
    return len(tok.encode(txt, add_special_tokens=False))

def balanced_sample(pool_df, n, seed):
    """Sample ~half from short-story and half from long-story buckets; compensate if one side is small."""
    rng = np.random.default_rng(seed)
    half = n // 2
    short_df = pool_df[pool_df["story_bucket"] == "S_STORY"]
    long_df  = pool_df[pool_df["story_bucket"] == "L_STORY"]

    take_s = min(half, len(short_df))
    take_l = min(n - take_s, len(long_df))
    if take_s + take_l < n:
        rem = n - (take_s + take_l)
        if len(short_df) - take_s >= len(long_df) - take_l:
            take_s = min(len(short_df), take_s + rem)
        else:
            take_l = min(len(long_df), take_l + rem)

    idx_s = rng.choice(short_df.index.values, size=take_s, replace=False) if take_s > 0 else []
    idx_l = rng.choice(long_df.index.values,  size=take_l, replace=False) if take_l > 0 else []
    chosen = pool_df.loc[list(idx_s) + list(idx_l)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return chosen

def tok_to_words(tokens, ratio=1.5):
    return max(1, int(round(tokens / ratio)))

# ---------------------- main ---------------------- #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="glnmario/news-qa-summarization")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--qa_n", type=int, default=50, help="QA items to select")
    ap.add_argument("--summ_short_n", type=int, default=50, help="Short summarization items to select")
    ap.add_argument("--summ_long_n", type=int, default=50, help="Long summarization items to select")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="./subset_plan")
    ap.add_argument("--tokenizer", type=str, default="gpt2")  # model-agnostic tokenizer
    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset: {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, split=args.split)

    cols = set(ds.column_names)
    story_col    = "story"    if "story" in cols else "document"
    question_col = "question" if "question" in cols else "questions"
    summary_col  = "summary"  if "summary" in cols else "abstract"
    # support 'answer' or 'answers'
    if "answer" in cols:
        answer_col = "answer"
    elif "answers" in cols:
        answer_col = "answers"
    else:
        raise ValueError(f"No answer column found in dataset. Columns: {ds.column_names}")

    print(f"[INFO] Resolving columns -> story={story_col}, question={question_col}, summary={summary_col}, answer={answer_col}")

    print(f"[INFO] Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print("[INFO] Computing token counts …")
    story_tok = [count_tokens(x, tok) for x in ds[story_col]]
    summ_tok  = [count_tokens(x, tok) for x in ds[summary_col]]
    ans_tok   = [count_tokens(x, tok) for x in ds[answer_col]]
    ques_txt  = [to_text(x) for x in ds[question_col]]

    df = pd.DataFrame({
        "row_id": np.arange(len(ds)),
        "story":   [to_text(x) for x in ds[story_col]],
        "question":[to_text(x) for x in ds[question_col]],
        "answer":  [to_text(x) for x in ds[answer_col]],
        "summary": [to_text(x) for x in ds[summary_col]],
        "story_tokens":   story_tok,
        "summary_tokens": summ_tok,
        "answer_tokens":  ans_tok,
    })

    # basic filtering
    df = df[(df["story_tokens"] > 0) & (df["summary_tokens"] > 0)].reset_index(drop=True)

    # thresholds
    P40 = int(np.percentile(df["summary_tokens"], 40))
    P60 = int(np.percentile(df["summary_tokens"], 60))
    P50_story = int(np.percentile(df["story_tokens"], 50))
    print(f"[INFO] Summary tokens percentiles: P40={P40}, P60={P60}")
    print(f"[INFO] Story tokens median P50={P50_story}")

    # buckets
    df["summ_bucket"] = np.where(df["summary_tokens"] <= P40, "SHORT",
                          np.where(df["summary_tokens"] >= P60, "LONG", "MID"))
    df["story_bucket"] = np.where(df["story_tokens"] <= P50_story, "S_STORY", "L_STORY")

    # targets from bucket medians (tokens -> words)
    short_median_tokens = int(np.median(df.loc[df["summ_bucket"] == "SHORT", "summary_tokens"])) if (df["summ_bucket"] == "SHORT").any() else 60
    long_median_tokens  = int(np.median(df.loc[df["summ_bucket"] == "LONG",  "summary_tokens"])) if (df["summ_bucket"] == "LONG").any()  else 240
    TARGET_WORDS_SHORT = tok_to_words(short_median_tokens)
    TARGET_WORDS_LONG  = tok_to_words(long_median_tokens)
    print(f"[INFO] TARGET_WORDS_SHORT ≈ {TARGET_WORDS_SHORT} (median {short_median_tokens} tokens)")
    print(f"[INFO] TARGET_WORDS_LONG  ≈ {TARGET_WORDS_LONG} (median {long_median_tokens} tokens)")

    # pools
    pool_short = df[df["summ_bucket"] == "SHORT"].copy()
    pool_long  = df[df["summ_bucket"] == "LONG"].copy()
    qa_pool    = df[(df["answer_tokens"] > 0) & (df["question"].apply(lambda s: len(s) > 0))].copy()

    # selection (balanced by story length)
    summ_short_sel = balanced_sample(pool_short, args.summ_short_n, seed=args.seed + 1)
    summ_long_sel  = balanced_sample(pool_long,  args.summ_long_n,  seed=args.seed + 2)
    qa_sel         = balanced_sample(qa_pool,    args.qa_n,         seed=args.seed + 3)

    # save helpers
    def save_csv(name, frame):
        dst = outdir / f"{name}.csv"
        keep_cols = [
            "row_id", "task", "story_bucket", "summ_bucket",
            "story_tokens", "summary_tokens", "answer_tokens",
            "story", "question", "answer", "summary", "target_words"
        ]
        # ensure columns exist
        if "task" not in frame.columns:
            frame["task"] = ""
        if "target_words" not in frame.columns:
            frame["target_words"] = np.nan
        frame[keep_cols].to_csv(dst, index=False)
        print(f"[OK] Saved {dst}")

    save_csv("qa_selected", qa_sel.assign(task="QA", target_words=np.nan))
    save_csv("summ_short_selected", summ_short_sel.assign(task="SUMM_SHORT", target_words=TARGET_WORDS_SHORT))
    save_csv("summ_long_selected",  summ_long_sel.assign(task="SUMM_LONG",  target_words=TARGET_WORDS_LONG))

    manifest = pd.concat([
        qa_sel.assign(task="QA", target_words=np.nan),
        summ_short_sel.assign(task="SUMM_SHORT", target_words=TARGET_WORDS_SHORT),
        summ_long_sel.assign(task="SUMM_LONG",  target_words=TARGET_WORDS_LONG),
    ], ignore_index=True).sample(frac=1.0, random_state=args.seed)

    manifest_dst = outdir / "manifest_all_tasks.csv"
    manifest.to_csv(manifest_dst, index=False)
    print(f"[OK] Saved {manifest_dst}")

    # write thresholds file
    with open(outdir / "length_targets.txt", "w", encoding="utf-8") as f:
        f.write(f"P40_summary_tokens={P40}\n")
        f.write(f"P60_summary_tokens={P60}\n")
        f.write(f"P50_story_tokens={P50_story}\n")
        f.write(f"TARGET_WORDS_SHORT={TARGET_WORDS_SHORT}  # from SHORT-bucket median\n")
        f.write(f"TARGET_WORDS_LONG={TARGET_WORDS_LONG}    # from LONG-bucket median\n")
        f.write("\nCounts:\n")
        f.write(f"QA selected: {len(qa_sel)}\n")
        f.write(f"Summ SHORT selected: {len(summ_short_sel)}\n")
        f.write(f"Summ LONG selected: {len(summ_long_sel)}\n")
    print(f"[OK] Saved {outdir / 'length_targets.txt'}")

if __name__ == "__main__":
    main()

