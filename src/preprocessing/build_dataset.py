#!/usr/bin/env python3
"""
Build paper-level dataset from ICLR review-level CSV.

Input:  data/raw/iclr_2025_detailed_reviews.csv
Output: data/processed/train.jsonl, val.jsonl, test.jsonl (+ dataset_info.json)

Each JSONL line:
{
  "paper_id": "...",
  "input_text": "...",
  "target_text": "DECISION: ...\nMETA_REVIEW:\n..."
}
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------
# Text + numeric helpers
# -------------------------

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def normalize_ws(s: str) -> str:
    return " ".join(s.split())


def trunc_chars(s: str, max_chars: int) -> str:
    s = normalize_ws(safe_str(s))
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def to_float_or_none(x) -> Optional[float]:
    s = safe_str(x)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def map_official_label_to_binary(label: str) -> Optional[str]:
    """
    Robustly map official_label -> ACCEPT/REJECT.
    Drops Pending/unknown.
    """
    s = safe_str(label)
    if not s:
        return None
    s_low = s.lower()

    if s_low.startswith("accept"):
        return "ACCEPT"
    if s_low.startswith("reject"):
        return "REJECT"
    # pending / desk-reject variants etc.
    return None


def jsonl_write(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@dataclass
class Truncation:
    title: int = 250
    abstract: int = 1200
    summary: int = 900
    strengths: int = 700
    weaknesses: int = 700
    questions: int = 500
    author_response: int = 600


# -------------------------
# Prompt construction
# -------------------------

def build_input_text(
    paper_id: str,
    title: str,
    abstract: str,
    reviews_df: pd.DataFrame,
    trunc: Truncation,
    include_score_changed: bool = False,
    include_rebuttal: bool = False,
    max_reviews: Optional[int] = None,
) -> str:
    """
    Build one structured prompt string for a paper.
    """
    df = reviews_df.copy()

    # numeric for sorting + aggregates
    df["final_rating_num"] = df.get("final_rating", pd.Series([None]*len(df))).apply(to_float_or_none)
    df["confidence_num"] = df.get("confidence", pd.Series([None]*len(df))).apply(to_float_or_none)

    # Sort: lowest rating first (negatives), then highest confidence
    df["rating_sort"] = df["final_rating_num"].fillna(9999.0)
    df["conf_sort"] = df["confidence_num"].fillna(-1.0)
    df = df.sort_values(["rating_sort", "conf_sort"], ascending=[True, False], kind="mergesort")

    if max_reviews is not None and len(df) > max_reviews:
        df = df.head(max_reviews)

    ratings = [x for x in df["final_rating_num"].tolist() if x is not None]
    confs = [x for x in df["confidence_num"].tolist() if x is not None]

    mean_rating = float(np.mean(ratings)) if ratings else None
    min_rating = float(np.min(ratings)) if ratings else None
    max_rating = float(np.max(ratings)) if ratings else None
    mean_conf = float(np.mean(confs)) if confs else None

    num_score_changed = None
    if include_score_changed and "score_changed" in df.columns:
        sc = df["score_changed"].astype(str).str.lower().str.strip()
        num_score_changed = int(sc.isin({"1", "true", "yes"}).sum())

    parts: List[str] = []

    # instruction header (consistent across all examples)
    parts.append(
        "TASK:\n"
        "Write an ICLR meta-review based on the paper and reviewer feedback. "
        "Also output a final decision.\n"
    )
    parts.append(
        "OUTPUT FORMAT:\n"
        "DECISION: <ACCEPT or REJECT>\n"
        "META_REVIEW:\n"
        "<text>\n"
    )

    parts.append(f"PAPER ID:\n{paper_id}\n")
    parts.append(f"PAPER TITLE:\n{trunc_chars(title, trunc.title) or 'N/A'}\n")
    parts.append(f"ABSTRACT:\n{trunc_chars(abstract, trunc.abstract) or 'N/A'}\n")

    # aggregates (cheap, useful signal)
    parts.append("AGGREGATES:")
    parts.append(f"- NumReviews: {len(df)}")
    if mean_rating is not None:
        parts.append(f"- MeanRating: {mean_rating:.2f}")
    if min_rating is not None and max_rating is not None:
        parts.append(f"- RatingRange: {min_rating:.0f} to {max_rating:.0f}")
    if mean_conf is not None:
        parts.append(f"- MeanConfidence: {mean_conf:.2f}")
    if include_score_changed and num_score_changed is not None:
        parts.append(f"- NumScoreChanged: {num_score_changed}")
    parts.append("")  # blank line

    parts.append("REVIEWS:\n")

    def get(row: pd.Series, col: str) -> str:
        return safe_str(row[col]) if col in row and pd.notna(row[col]) else ""

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        parts.append(f"REVIEW {i}:")

        # numeric / categorical fields
        for field in ["final_rating", "confidence", "soundness", "presentation", "contribution"]:
            v = get(r, field)
            if v:
                parts.append(f"{field.replace('_', ' ').title().replace(' ', '')}: {v}")

        if include_score_changed and "score_changed" in df.columns:
            sc = get(r, "score_changed").lower()
            if sc in {"1", "true", "yes"}:
                parts.append("ScoreChanged: Yes")
            elif sc in {"0", "false", "no"}:
                parts.append("ScoreChanged: No")

        # text fields
        summary = trunc_chars(get(r, "summary"), trunc.summary)
        strengths = trunc_chars(get(r, "strengths"), trunc.strengths)
        weaknesses = trunc_chars(get(r, "weaknesses"), trunc.weaknesses)
        questions = trunc_chars(get(r, "questions"), trunc.questions)

        if summary:
            parts.append(f"Summary:\n{summary}")
        if strengths:
            parts.append(f"Strengths:\n{strengths}")
        if weaknesses:
            parts.append(f"Weaknesses:\n{weaknesses}")
        if questions:
            parts.append(f"Questions:\n{questions}")

        if include_rebuttal and "author_response_to_this_review" in df.columns:
            ar = trunc_chars(get(r, "author_response_to_this_review"), trunc.author_response)
            if ar:
                parts.append(f"AuthorResponse:\n{ar}")

        parts.append("")  # blank line between reviews

    return "\n".join(parts).strip() + "\n"


def build_target_text(decision: str, meta_review: str) -> str:
    """
    Target string for seq2seq: decision + meta review text.
    """
    return f"DECISION: {decision}\nMETA_REVIEW:\n{safe_str(meta_review)}\n"


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/raw/iclr_2025_detailed_reviews.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)

    # splits are fractions of total papers
    parser.add_argument("--test_size", type=float, default=0.10)
    parser.add_argument("--val_size", type=float, default=0.10)

    # experiment toggles (start false for run1)
    parser.add_argument("--include_score_changed", action="store_true")
    parser.add_argument("--include_rebuttal", action="store_true")

    parser.add_argument("--max_reviews", type=int, default=0, help="0 means no limit")

    # truncation knobs (characters)
    parser.add_argument("--title_chars", type=int, default=250)
    parser.add_argument("--abstract_chars", type=int, default=1200)
    parser.add_argument("--summary_chars", type=int, default=900)
    parser.add_argument("--strengths_chars", type=int, default=700)
    parser.add_argument("--weaknesses_chars", type=int, default=700)
    parser.add_argument("--questions_chars", type=int, default=500)
    parser.add_argument("--author_response_chars", type=int, default=600)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "paper_id" not in df.columns:
        raise ValueError("CSV must contain a 'paper_id' column.")

    # decision mapping
    if "official_label" not in df.columns:
        raise ValueError("CSV must contain an 'official_label' column.")
    df["decision"] = df["official_label"].apply(map_official_label_to_binary)

    # drop unknown/pending
    df = df[df["decision"].notna()].copy()

    # meta_review must exist (paper-level)
    if "meta_review" not in df.columns:
        raise ValueError("CSV must contain a 'meta_review' column.")
    df["meta_review_clean"] = df["meta_review"].astype(str).str.strip()
    df = df[df["meta_review_clean"].str.len() > 0].copy()
    df = df[df["meta_review_clean"].str.lower().ne("nan")].copy()

    # truncation config
    trunc = Truncation(
        title=args.title_chars,
        abstract=args.abstract_chars,
        summary=args.summary_chars,
        strengths=args.strengths_chars,
        weaknesses=args.weaknesses_chars,
        questions=args.questions_chars,
        author_response=args.author_response_chars,
    )

    max_reviews = None if args.max_reviews == 0 else args.max_reviews

    # group -> one example per paper
    examples: List[Dict] = []
    grouped = df.groupby("paper_id", sort=False)

    for paper_id, g in grouped:
        # paper fields (take first)
        title = safe_str(g["title"].iloc[0]) if "title" in g.columns else ""
        abstract = safe_str(g["abstract"].iloc[0]) if "abstract" in g.columns else ""

        decision = safe_str(g["decision"].iloc[0])
        meta_review = safe_str(g["meta_review_clean"].iloc[0])

        input_text = build_input_text(
            paper_id=str(paper_id),
            title=title,
            abstract=abstract,
            reviews_df=g,
            trunc=trunc,
            include_score_changed=args.include_score_changed,
            include_rebuttal=args.include_rebuttal,
            max_reviews=max_reviews,
        )
        target_text = build_target_text(decision=decision, meta_review=meta_review)

        examples.append(
            {
                "paper_id": str(paper_id),
                "input_text": input_text,
                "target_text": target_text,
                # optional convenience field:
                "decision": decision,
            }
        )

    if not examples:
        raise RuntimeError("No paper-level examples were produced. Check label mapping / filters.")

    # split by paper_id, stratified by decision
    labels = [ex["decision"] for ex in examples]
    idxs = np.arange(len(examples))

    train_idxs, test_idxs = train_test_split(
        idxs,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    # val_size is fraction of total; convert to fraction of remaining
    remaining = train_idxs
    remaining_labels = [examples[i]["decision"] for i in remaining]
    val_fraction_of_remaining = args.val_size / (1.0 - args.test_size)

    train2_idxs, val_idxs = train_test_split(
        remaining,
        test_size=val_fraction_of_remaining,
        random_state=args.seed,
        stratify=remaining_labels,
    )

    def take(id_list) -> List[Dict]:
        return [examples[i] for i in id_list]

    train_rows = take(train2_idxs)
    val_rows = take(val_idxs)
    test_rows = take(test_idxs)

    # write outputs
    jsonl_write(out_dir / "train.jsonl", train_rows)
    jsonl_write(out_dir / "val.jsonl", val_rows)
    jsonl_write(out_dir / "test.jsonl", test_rows)

    info = {
        "csv_path": str(csv_path),
        "num_papers_total": len(examples),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "include_score_changed": bool(args.include_score_changed),
        "include_rebuttal": bool(args.include_rebuttal),
        "max_reviews": max_reviews,
        "truncation_chars": trunc.__dict__,
        "label_distribution_total": {
            k: int(v) for k, v in pd.Series(labels).value_counts().items()
        },
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("✅ Wrote dataset:")
    print(f"  {out_dir / 'train.jsonl'}  ({len(train_rows)} papers)")
    print(f"  {out_dir / 'val.jsonl'}    ({len(val_rows)} papers)")
    print(f"  {out_dir / 'test.jsonl'}   ({len(test_rows)} papers)")
    print(f"  {out_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
