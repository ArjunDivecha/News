#!/usr/bin/env python3
"""
Build the expert-review workbook for the News ETF classifier v2 training round.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from expert_review_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_UNIVERSE_PATH,
    add_review_workbook_formatting,
    build_user_prompt,
    clean_text,
    demo_predictions,
    parse_model_json,
    stratified_sample,
    tags_to_string,
    universe_to_model_input,
)


DEFAULT_MODEL_PATH = "tinker://3a0f33d5-93be-5c30-b756-c183909f1804:train:0/sampler_weights/final"
DEFAULT_SECOND_LABELER_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLAMA_CACHE = Path(__file__).resolve().parents[1] / "outputs" / "comparison" / "ETF_Master_List_FineTuned.xlsx"


def try_load_1password_credentials(names: list[str]) -> None:
    missing = [name for name in names if not os.environ.get(f"{name.upper()}_API_KEY")]
    if not missing:
        return
    try:
        sys.path.insert(0, str(Path.home() / "python_utils"))
        from onepassword_credentials import load_credentials

        load_credentials(missing, verbose=False)
    except Exception:
        return


def load_existing_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_predict_funds_module():
    script_path = Path(__file__).with_name("predict_funds.py")
    spec = importlib.util.spec_from_file_location("predict_funds", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ticker_root(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return text.split()[0].split(".")[0]


def load_cached_llama_predictions(universe: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        raise FileNotFoundError(f"Llama cache not found: {cache_path}")
    cache = pd.read_excel(cache_path)
    required = {"llama_tier1", "llama_tier2", "llama_tier3"}
    missing = required - set(cache.columns)
    if missing:
        raise ValueError(f"Llama cache is missing columns: {sorted(missing)}")

    key_col = "Ticker" if "Ticker" in cache.columns else "Bloomberg"
    cache = cache.copy()
    cache["_ticker_root"] = cache[key_col].map(ticker_root)
    cache = cache.drop_duplicates("_ticker_root", keep="last")
    by_root = cache.set_index("_ticker_root")

    rows: list[dict[str, Any]] = []
    fallback_count = 0
    for _, row in universe.iterrows():
        root = ticker_root(row["yf_ticker"])
        if root in by_root.index:
            pred = by_root.loc[root]
            llama_tier1 = pred.get("llama_tier1")
            llama_tier2 = pred.get("llama_tier2")
            llama_tags = pred.get("llama_tier3")
            raw = "cached_comparison_workbook"
        else:
            fallback_count += 1
            llama_tier1 = row.get("tier1")
            llama_tier2 = row.get("tier2")
            llama_tags = row.get("tags")
            raw = "fallback_current_universe_label"
        rows.append(
            {
                "yf_ticker": clean_text(row["yf_ticker"]),
                "llama_tier1": clean_text(llama_tier1),
                "llama_tier2": clean_text(llama_tier2),
                "llama_tier3_tags": tags_to_string(llama_tags),
                "llama_raw_response": raw,
            }
        )
    print(f"Loaded cached Llama predictions from {cache_path}")
    print(f"Cached Llama fallback rows: {fallback_count} of {len(universe)}")
    return pd.DataFrame(rows)


def run_llama_predictions(universe: pd.DataFrame, model_path: str, max_tokens: int) -> pd.DataFrame:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set. Use --demo-predictions for a no-API smoke run.")

    predict_funds = load_predict_funds_module()
    import tinker

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    model_input = universe_to_model_input(universe)

    rows: list[dict[str, Any]] = []
    for idx, row in tqdm(model_input.iterrows(), total=len(model_input), desc="Llama"):
        result = predict_funds.classify_fund(sampling_client, predict_funds.build_prompt(row), max_tokens=max_tokens)
        rows.append(
            {
                "yf_ticker": clean_text(universe.iloc[idx]["yf_ticker"]),
                "llama_tier1": result.get("tier_1", "Unknown"),
                "llama_tier2": result.get("tier_2", "Unknown"),
                "llama_tier3_tags": tags_to_string(result.get("tier_3", [])),
                "llama_raw_response": "",
            }
        )
    return pd.DataFrame(rows)


def run_deepseek_predictions(
    universe: pd.DataFrame,
    model: str,
    base_url: str,
    max_tokens: int,
    checkpoint_path: Path,
    workers: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Use --demo-predictions for a no-API smoke run.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    rows: list[dict[str, Any]] = []
    done: set[str] = set()

    if checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path)
        rows = existing.to_dict("records")
        done = set(existing["yf_ticker"].map(clean_text))
        print(f"Resuming DeepSeek from {checkpoint_path}: {len(done)} completed rows")

    pending = [(clean_text(row["yf_ticker"]), row.copy()) for _, row in universe.iterrows() if clean_text(row["yf_ticker"]) not in done]

    def classify_one(ticker: str, row: pd.Series) -> tuple[dict[str, Any], dict[str, int]]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_completion_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SECOND_LABELER_SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(row)},
                    ],
                )
                response_text = response.choices[0].message.content.strip()
                parsed = parse_model_json(response_text)
                usage_row = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                }
                return (
                    {
                        "yf_ticker": ticker,
                        "deepseek_tier1": parsed["tier1"],
                        "deepseek_tier2": parsed["tier2"],
                        "deepseek_tier3_tags": tags_to_string(parsed["tier3_tags"]),
                        "deepseek_raw_response": response_text,
                    },
                    usage_row,
                )
            except Exception as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"DeepSeek failed for {ticker} after 3 attempts: {last_error}") from last_error

    def save_checkpoint() -> None:
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        pd.DataFrame(rows).to_parquet(tmp_path, index=False)
        tmp_path.replace(checkpoint_path)

    if not pending:
        return pd.DataFrame(rows), usage

    print(f"Running DeepSeek with {workers} concurrent workers")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_one, ticker, row): ticker for ticker, row in pending}
        with tqdm(total=len(universe), initial=len(done), desc="DeepSeek") as progress:
            for future in as_completed(futures):
                result_row, usage_row = future.result()
                rows.append(result_row)
                done.add(result_row["yf_ticker"])
                usage["prompt_tokens"] += usage_row["prompt_tokens"]
                usage["completion_tokens"] += usage_row["completion_tokens"]
                save_checkpoint()
                progress.update(1)

    order = {clean_text(ticker): idx for idx, ticker in enumerate(universe["yf_ticker"])}
    return pd.DataFrame(rows).sort_values("yf_ticker", key=lambda s: s.map(order)).reset_index(drop=True), usage


SECOND_LABELER_SYSTEM_PROMPT = """You are an expert ETF classification specialist. Classify each ETF into the existing News taxonomy.

TIER-1 CATEGORIES (pick exactly one):
- Equities
- Fixed Income
- Commodities
- Currencies (FX)
- Multi-Asset / Thematic
- Volatility / Risk Premia
- Alternative / Synthetic
- Factor

TIER-2 CATEGORIES (examples):
- Equities: Global Indices | Sector Indices | Country/Regional | Thematic/Factor | Real Estate / REITs
- Fixed Income: Sovereign Bonds | Corporate Credit | Credit Spreads | Yield Curves | Broad Fixed Income
- Commodities: Energy | Metals | Agriculture | Broad Commodities
- Currencies: Majors | EM FX | Broad Currency
- Multi-Asset / Thematic: Cross-Asset Indices | Inflation/Growth Themes | Thematic Baskets
- Volatility / Risk Premia: Vol Indices | Carry/Value Factors | Risk Premia Strategies
- Alternative / Synthetic: Quant/Style Baskets | Custom/Proprietary | Factor-Based
- Factor: Equity Factor | Rates Factor | Credit Factor | Commodity Factor | FX Factor | Volatility Factor

TIER-3 TAGS: choose a concise list using ONLY these exact strings. Do not invent
synonyms, narrower variants, or spelling variants. If the exact tag is not
available, omit it.

Allowed tags:
- Equity
- Credit
- FX
- Commodity
- Multi-Asset
- Volatility
- Alternative
- US
- Europe
- Asia
- EM
- Global
- China
- Japan
- India
- Canada
- UAE
- APAC
- Australia
- Developed
- Tech
- Energy
- Financials
- Healthcare
- Industrials
- Consumer
- Defensive
- ESG
- Dividend
- Growth
- Value
- Momentum
- Quality
- Infrastructure
- Real Estate
- Utilities
- Materials
- Active
- Passive
- Equal-Weight
- Thematic
- Quantitative
- Options-Based
- Factor-Based
- Low Volatility
- Domestic
- International
- Diversified
- Investment Grade
- High Yield
- Short Duration
- Long Duration
- IG Credit
- HY Credit

Return only JSON in this shape:
{"ticker":"TICKER","tier1":"Category","tier2":"Sub-category","tier3_tags":["tag1","tag2"]}"""


def warn_cost(universe: pd.DataFrame, ceiling: float) -> None:
    estimated_input_tokens = int(universe["description"].fillna("").str.len().sum() / 4) + len(universe) * 350
    estimated_output_tokens = len(universe) * 120
    estimated_cost = estimated_input_tokens / 1_000_000 * 0.55 + estimated_output_tokens / 1_000_000 * 3.48
    print(f"Estimated DeepSeek usage: input={estimated_input_tokens:,}, output={estimated_output_tokens:,}, rough_cost=${estimated_cost:.2f}")
    if estimated_cost > ceiling:
        print(f"WARNING: estimated DeepSeek cost exceeds configured ceiling ${ceiling:.2f}; continuing by default.")


def build_review_frames(universe: pd.DataFrame, predictions: pd.DataFrame, control_size: int, tier3_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = universe.merge(predictions, on="yf_ticker", how="left", validate="one_to_one")
    merged["description_short"] = merged["description"].fillna("").astype(str).str.slice(0, 300)
    merged["tier1_disagree"] = merged["llama_tier1"].fillna("") != merged["deepseek_tier1"].fillna("")
    merged["tier2_disagree"] = merged["llama_tier2"].fillna("") != merged["deepseek_tier2"].fillna("")
    merged["is_disagreement"] = merged["tier1_disagree"] | merged["tier2_disagree"]

    disagreement = merged[merged["is_disagreement"]].copy()
    agreement = merged[~merged["is_disagreement"]].copy()
    controls = stratified_sample(agreement, "llama_tier1", control_size, min_per_group=3, seed=seed)

    disagreement["batch_type"] = "disagreement"
    controls["batch_type"] = "control"
    review = pd.concat([disagreement, controls], ignore_index=True)
    review = review.sample(frac=1, random_state=seed).reset_index(drop=True)
    review["expert_tier1"] = ""
    review["expert_tier2"] = ""
    review["expert_notes"] = ""
    review["verdict"] = ""

    adjudicate_cols = [
        "yf_ticker",
        "name",
        "description_short",
        "batch_type",
        "llama_tier1",
        "llama_tier2",
        "deepseek_tier1",
        "deepseek_tier2",
        "expert_tier1",
        "expert_tier2",
        "expert_notes",
        "verdict",
    ]
    adjudicate = review[adjudicate_cols].rename(columns={"description_short": "description"})

    spotcheck_source = stratified_sample(merged, "llama_tier1", tier3_size, min_per_group=3, seed=seed + 1)
    spotcheck = spotcheck_source[
        ["yf_ticker", "name", "description_short", "llama_tier3_tags", "deepseek_tier3_tags"]
    ].rename(columns={"description_short": "description"})
    spotcheck["expert_tier3_tags"] = ""

    return adjudicate, spotcheck


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expert-review workbook for ETF classifier labels.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--llama-cache", type=Path, default=DEFAULT_LLAMA_CACHE, help="Existing workbook with llama_tier columns.")
    parser.add_argument("--use-llama-cache", action="store_true", help="Use cached Llama predictions instead of Tinker inference.")
    parser.add_argument("--second-labeler-model", default=os.getenv("EXPERT_REVIEW_SECOND_LABELER_MODEL", DEFAULT_SECOND_LABELER_MODEL))
    parser.add_argument("--deepseek-base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL))
    parser.add_argument("--control-size", type=int, default=80)
    parser.add_argument("--tier3-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--deepseek-workers", type=int, default=6)
    parser.add_argument("--deepseek-cost-ceiling", type=float, default=25.0)
    parser.add_argument("--demo-predictions", action="store_true", help="Use deterministic fake predictions for local smoke testing.")
    parser.add_argument("--refresh", action="store_true", help="Ignore raw_predictions.parquet and regenerate predictions.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not args.demo_predictions:
        try_load_1password_credentials(["DeepSeek"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_predictions.parquet"
    deepseek_checkpoint_path = args.output_dir / "deepseek_predictions_partial.parquet"
    review_path = args.output_dir / "review_batch.xlsx"

    universe = pd.read_excel(args.universe)
    if "yf_ticker" not in universe.columns:
        raise SystemExit("Input universe must contain yf_ticker.")
    universe["yf_ticker"] = universe["yf_ticker"].map(clean_text)

    predictions = None if args.refresh else load_existing_cache(raw_path)
    if predictions is None:
        if args.demo_predictions:
            predictions = demo_predictions(universe)
        else:
            warn_cost(universe, args.deepseek_cost_ceiling)
            if args.use_llama_cache:
                llama = load_cached_llama_predictions(universe, args.llama_cache)
            else:
                llama = run_llama_predictions(universe, args.model_path, args.max_tokens)
            deepseek, usage = run_deepseek_predictions(
                universe,
                args.second_labeler_model,
                args.deepseek_base_url,
                args.max_tokens,
                deepseek_checkpoint_path,
                args.deepseek_workers,
            )
            print(f"Actual DeepSeek usage: prompt={usage['prompt_tokens']:,}, completion={usage['completion_tokens']:,}")
            predictions = llama.merge(deepseek, on="yf_ticker", how="outer", validate="one_to_one")
        predictions.to_parquet(raw_path, index=False)
        print(f"Wrote {raw_path}")
    else:
        print(f"Loaded cached predictions from {raw_path}")

    adjudicate, spotcheck = build_review_frames(universe, predictions, args.control_size, args.tier3_size, args.seed)
    with pd.ExcelWriter(review_path, engine="openpyxl") as writer:
        adjudicate.to_excel(writer, sheet_name="adjudicate", index=False)
        spotcheck.to_excel(writer, sheet_name="tier3_spotcheck", index=False)
    add_review_workbook_formatting(review_path, len(adjudicate), len(spotcheck))

    print(f"Wrote {review_path}")
    print(f"Adjudication rows: {len(adjudicate)}")
    print(adjudicate["batch_type"].value_counts().to_string())
    print(f"Tier3 spot-check rows: {len(spotcheck)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
