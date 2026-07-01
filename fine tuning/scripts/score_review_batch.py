#!/usr/bin/env python3
"""
Score a completed or partially completed expert-review workbook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from expert_review_common import DEFAULT_OUTPUT_DIR, micro_f1, parse_tags, pct


def accuracy(df: pd.DataFrame, prefix: str, tier: str) -> float:
    if df.empty:
        return 0.0
    return pct((df[f"{prefix}_{tier}"] == df[f"expert_{tier}"]).sum(), len(df))


def scored_adjudicate(adjudicate: pd.DataFrame, second_prefix: str) -> pd.DataFrame:
    required = adjudicate.dropna(subset=["expert_tier1", "expert_tier2"], how="any").copy()
    for col in ["expert_tier1", "expert_tier2", "llama_tier1", "llama_tier2", f"{second_prefix}_tier1", f"{second_prefix}_tier2", "verdict", "batch_type"]:
        required[col] = required[col].fillna("").astype(str).str.strip()
    required = required[(required["expert_tier1"] != "") & (required["expert_tier2"] != "")]
    return required


def tier3_metrics(spotcheck: pd.DataFrame, prefix: str) -> dict[str, float]:
    rows = spotcheck.dropna(subset=["expert_tier3_tags"]).copy()
    rows["expert_tier3_tags"] = rows["expert_tier3_tags"].fillna("").astype(str).str.strip()
    rows = rows[rows["expert_tier3_tags"] != ""]
    if rows.empty:
        return {"n": 0, "exact": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    predicted = [set(parse_tags(x)) for x in rows[f"{prefix}_tier3_tags"]]
    actual = [set(parse_tags(x)) for x in rows["expert_tier3_tags"]]
    exact = sum({t.lower() for t in p} == {t.lower() for t in a} for p, a in zip(predicted, actual))
    f1 = micro_f1(predicted, actual)
    return {"n": len(rows), "exact": pct(exact, len(rows)), **f1}


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_report(workbook: Path, output_path: Path) -> None:
    adjudicate = pd.read_excel(workbook, sheet_name="adjudicate")
    spotcheck = pd.read_excel(workbook, sheet_name="tier3_spotcheck")
    second_prefix = "deepseek" if "deepseek_tier1" in adjudicate.columns else "haiku"
    second_label = "DeepSeek" if second_prefix == "deepseek" else "Haiku"
    scored = scored_adjudicate(adjudicate, second_prefix)

    lines = [
        "# Expert Review Report",
        "",
        f"Workbook: `{workbook}`",
        f"Completed adjudication rows scored: {len(scored)}",
        "",
        "## Tier1/Tier2 Accuracy vs Expert",
        "",
    ]

    summary_rows: list[list[str]] = []
    for batch_type in ["disagreement", "control"]:
        subset = scored[scored["batch_type"] == batch_type]
        summary_rows.append(
            [
                batch_type,
                str(len(subset)),
                f"{accuracy(subset, 'llama', 'tier1'):.1%}",
                f"{accuracy(subset, 'llama', 'tier2'):.1%}",
                f"{accuracy(subset, second_prefix, 'tier1'):.1%}",
                f"{accuracy(subset, second_prefix, 'tier2'):.1%}",
                f"{pct((subset['verdict'] == 'both_wrong').sum(), len(subset)):.1%}" if len(subset) else "0.0%",
            ]
        )
    lines.append(
        markdown_table(
            ["Batch", "Rows", "Llama Tier1", "Llama Tier2", f"{second_label} Tier1", f"{second_label} Tier2", "Both Wrong Verdict"],
            summary_rows,
        )
    )

    lines.extend(["", "## Per-Tier1 Breakdown", ""])
    breakdown_rows: list[list[str]] = []
    for tier1, group in scored.groupby("expert_tier1"):
        breakdown_rows.append(
            [
                tier1,
                str(len(group)),
                f"{accuracy(group, 'llama', 'tier1'):.1%}",
                f"{accuracy(group, 'llama', 'tier2'):.1%}",
                f"{accuracy(group, second_prefix, 'tier1'):.1%}",
                f"{accuracy(group, second_prefix, 'tier2'):.1%}",
            ]
        )
    lines.append(markdown_table(["Expert Tier1", "Rows", "Llama T1", "Llama T2", f"{second_label} T1", f"{second_label} T2"], breakdown_rows or [["None yet", "0", "0.0%", "0.0%", "0.0%", "0.0%"]]))

    llama_tags = tier3_metrics(spotcheck, "llama")
    second_tags = tier3_metrics(spotcheck, second_prefix)
    lines.extend(["", "## Tier3 Tags Metric Check", ""])
    lines.append(
        markdown_table(
            ["Model", "Rows", "Exact Set Match", "Micro Precision", "Micro Recall", "Micro F1"],
            [
                ["Llama", str(llama_tags["n"]), f"{llama_tags['exact']:.1%}", f"{llama_tags['precision']:.1%}", f"{llama_tags['recall']:.1%}", f"{llama_tags['f1']:.1%}"],
                [second_label, str(second_tags["n"]), f"{second_tags['exact']:.1%}", f"{second_tags['precision']:.1%}", f"{second_tags['recall']:.1%}", f"{second_tags['f1']:.1%}"],
            ],
        )
    )

    max_f1 = max(llama_tags["f1"], second_tags["f1"])
    max_exact = max(llama_tags["exact"], second_tags["exact"])
    if max_f1 >= 0.80 and max_exact <= 0.65:
        decision = "Tier3 exact-match weakness looks like a metric artifact; switch evaluation to per-tag F1 before spending label time on tags."
    elif max_f1 == 0 and max_exact == 0:
        decision = "Tier3 decision pending; no completed expert tag rows were available."
    else:
        decision = "Tier3 may be genuinely weak; consider a dedicated tag-labeling round if this persists after more completed rows."
    lines.extend(["", "## Decision Notes", "", f"- {decision}"])

    control = scored[scored["batch_type"] == "control"]
    both_wrong_control = pct((control["verdict"] == "both_wrong").sum(), len(control)) if len(control) else 0.0
    lines.append(f"- Control-set both-wrong rate: {both_wrong_control:.1%}.")
    if both_wrong_control > 0.05:
        lines.append("- Shared model errors are material on the control set; agreement between machine labelers was hiding real errors.")
    elif len(control):
        lines.append("- Shared model errors do not look material in the completed control rows so far.")
    else:
        lines.append("- Control-set assessment pending; no completed control rows were available.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an expert-review workbook.")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_OUTPUT_DIR / "review_batch.xlsx")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "review_report.md")
    args = parser.parse_args()
    write_report(args.workbook, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
