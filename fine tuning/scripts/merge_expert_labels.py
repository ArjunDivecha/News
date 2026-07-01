#!/usr/bin/env python3
"""
Merge expert corrections into machine labels and emit train/test JSONL splits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from expert_review_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_UNIVERSE_PATH,
    build_training_example,
    clean_text,
    parse_tags,
    write_jsonl,
)


def load_completed_adjudications(workbook: Path) -> pd.DataFrame:
    adjudicate = pd.read_excel(workbook, sheet_name="adjudicate")
    adjudicate["yf_ticker"] = adjudicate["yf_ticker"].map(clean_text)
    for col in ["expert_tier1", "expert_tier2"]:
        adjudicate[col] = adjudicate[col].fillna("").astype(str).str.strip()
    return adjudicate[(adjudicate["expert_tier1"] != "") & (adjudicate["expert_tier2"] != "")].copy()


def stratified_expert_test(adjudicated: pd.DataFrame, test_ratio: float, seed: int) -> set[str]:
    rng = random.Random(seed)
    picks: set[str] = set()
    for _, group in adjudicated.groupby("expert_tier1"):
        tickers = group["yf_ticker"].tolist()
        rng.shuffle(tickers)
        n = max(1, round(len(tickers) * test_ratio)) if len(tickers) >= 2 else len(tickers)
        picks.update(tickers[:n])
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge expert labels and create expert train/test JSONL.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_OUTPUT_DIR / "review_batch.xlsx")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT_DIR / "raw_predictions.parquet")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    universe = pd.read_excel(args.universe)
    universe["yf_ticker"] = universe["yf_ticker"].map(clean_text)
    predictions = pd.read_parquet(args.predictions)
    predictions["yf_ticker"] = predictions["yf_ticker"].map(clean_text)
    adjudicated = load_completed_adjudications(args.workbook)

    merged = universe.merge(predictions, on="yf_ticker", how="left", validate="one_to_one")
    expert_map = adjudicated.set_index("yf_ticker")[["expert_tier1", "expert_tier2"]].to_dict("index")

    examples = []
    expert_test_tickers = stratified_expert_test(adjudicated, args.test_ratio, args.seed) if not adjudicated.empty else set()

    for _, row in merged.iterrows():
        ticker = clean_text(row["yf_ticker"])
        if ticker in expert_map:
            tier1 = expert_map[ticker]["expert_tier1"]
            tier2 = expert_map[ticker]["expert_tier2"]
        else:
            tier1 = row.get("llama_tier1") or row.get("tier1")
            tier2 = row.get("llama_tier2") or row.get("tier2")
        tags = row.get("llama_tier3_tags") if clean_text(row.get("llama_tier3_tags")) else row.get("tags")
        examples.append((ticker, build_training_example(row, tier1, tier2, parse_tags(tags))))

    train_examples = [example for ticker, example in examples if ticker not in expert_test_tickers]
    test_examples = [example for ticker, example in examples if ticker in expert_test_tickers]

    train_count = write_jsonl(args.output_dir / "train_expert.jsonl", train_examples)
    test_count = write_jsonl(args.output_dir / "test_expert.jsonl", test_examples)

    manifest = {
        "universe": str(args.universe),
        "workbook": str(args.workbook),
        "predictions": str(args.predictions),
        "completed_expert_rows": int(len(adjudicated)),
        "expert_test_tickers": sorted(expert_test_tickers),
        "train_expert_rows": train_count,
        "test_expert_rows": test_count,
        "note": "To retrain with 05_train_proper.py, copy or symlink train_expert.jsonl to data/processed/train.jsonl and provide a validation split.",
    }
    (args.output_dir / "expert_label_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_dir / 'train_expert.jsonl'} ({train_count} rows)")
    print(f"Wrote {args.output_dir / 'test_expert.jsonl'} ({test_count} rows)")
    print(f"Wrote {args.output_dir / 'expert_label_manifest.json'}")


if __name__ == "__main__":
    main()
