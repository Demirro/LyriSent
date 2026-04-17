"""
Recalculate estimated_cost_usd in comparison_results/**/runtime_cost_summary.csv
from prompt_tokens + completion_tokens using published per-1M token prices.

Default: gpt-5.4 short context ($2.50 / $15.00 per 1M in/out).
Use --gpt54-tier long for long-context rates ($5.00 / $22.50 per 1M).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "comparison_results"

# USD per 1M tokens (user-provided)
GPT54_SHORT = {"input": 2.50, "output": 15.00}
GPT54_LONG = {"input": 5.00, "output": 22.50}
GPT4O = {"input": 2.50, "cached_input": 1.25, "output": 10.00}


def _f(x: str | None) -> float:
    if x is None or str(x).strip() == "":
        return 0.0
    try:
        return float(x)
    except ValueError:
        return 0.0


def estimate_usd(model: str, prompt_tokens: float, completion_tokens: float, *, gpt54_tier: str) -> float | None:
    name = (model or "").strip().lower()
    pt = max(0.0, prompt_tokens)
    ct = max(0.0, completion_tokens)
    if "gpt-5.4" in name or name.startswith("gpt-5"):
        p = GPT54_LONG if gpt54_tier == "long" else GPT54_SHORT
        return (pt / 1_000_000.0) * p["input"] + (ct / 1_000_000.0) * p["output"]
    if "gpt-4o" in name:
        return (pt / 1_000_000.0) * GPT4O["input"] + (ct / 1_000_000.0) * GPT4O["output"]
    return None


def process_file(path: Path, *, gpt54_tier: str, dry_run: bool) -> bool:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "estimated_cost_usd" not in fieldnames:
        print(f"Skip (no estimated_cost_usd column): {path}")
        return False

    updated = 0
    for row in rows:
        if (row.get("method") or "").strip() != "OpenAI":
            continue
        model = row.get("model") or ""
        pt = _f(row.get("prompt_tokens"))
        ct = _f(row.get("completion_tokens"))
        new_cost = estimate_usd(model, pt, ct, gpt54_tier=gpt54_tier)
        if new_cost is None:
            print(f"  Warning: unknown model {model!r} in {path.name}, leaving cost unchanged")
            continue
        row["estimated_cost_usd"] = f"{new_cost:.6f}"
        updated += 1

    if updated == 0:
        print(f"Skip (no OpenAI rows updated): {path}")
        return False

    if dry_run:
        print(f"DRY RUN would write: {path}")
        for row in rows:
            if (row.get("method") or "").strip() == "OpenAI":
                print(f"  {row.get('method_variant')}: estimated_cost_usd={row.get('estimated_cost_usd')}")
        return True

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated ({updated} OpenAI row(s)): {path}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gpt54-tier",
        choices=("short", "long"),
        default="short",
        help="gpt-5.4 pricing tier (default: short)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not RESULTS_DIR.is_dir():
        print(f"No {RESULTS_DIR}")
        return

    files = sorted(RESULTS_DIR.rglob("runtime_cost_summary.csv"))
    if not files:
        print(f"No runtime_cost_summary.csv under {RESULTS_DIR}")
        return

    for p in files:
        process_file(p, gpt54_tier=args.gpt54_tier, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
