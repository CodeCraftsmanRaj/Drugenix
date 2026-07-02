from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from drugenix.pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drugenix",
        description="Drugenix: PubMed retrieval + LLM reasoning + Validation Agent for drug repurposing.",
    )
    parser.add_argument(
        "disease_query",
        type=str,
        help="Disease or condition to search (e.g., 'Alzheimer disease').",
    )
    parser.add_argument(
        "--retmax",
        type=int,
        default=40,
        help="How many PubMed records to fetch before reranking (default: 40).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many retrieved abstracts to pass to reasoning (default: 8).",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save full output JSON.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="",
        help="LLM provider: openai | gemini | ollama | custom (openai-compatible).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name for selected provider.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Override provider base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Optional API key override for selected provider.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        default=False,
        help="Skip the Validation Agent and show raw reasoning output only.",
    )
    return parser


def _print_section(title: str, char: str = "=") -> None:
    width = 60
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Configure LLM provider from CLI flags
    # ------------------------------------------------------------------
    if args.provider:
        provider = args.provider.strip().lower()
        os.environ["LLM_PROVIDER"] = provider

        if provider in {"openai", "open-ai"}:
            if args.api_key:
                os.environ["OPENAI_API_KEY"] = args.api_key
            if args.model:
                os.environ["OPENAI_MODEL"] = args.model
            if args.base_url:
                os.environ["OPENAI_BASE_URL"] = args.base_url
        elif provider in {"gemini", "gemmini", "google"}:
            if args.api_key:
                os.environ["GEMINI_API_KEY"] = args.api_key
            if args.model:
                os.environ["GEMINI_MODEL"] = args.model
            if args.base_url:
                os.environ["GEMINI_BASE_URL"] = args.base_url
        elif provider == "ollama":
            if args.api_key:
                os.environ["OLLAMA_API_KEY"] = args.api_key
            if args.model:
                os.environ["OLLAMA_MODEL"] = args.model
            if args.base_url:
                os.environ["OLLAMA_BASE_URL"] = args.base_url
        else:
            if args.api_key:
                os.environ["LLM_API_KEY"] = args.api_key
            if args.model:
                os.environ["LLM_MODEL"] = args.model
            if args.base_url:
                os.environ["LLM_BASE_URL"] = args.base_url

    # ------------------------------------------------------------------
    # Run the full pipeline
    # ------------------------------------------------------------------
    result = run_pipeline(
        disease_query=args.disease_query,
        retmax=args.retmax,
        top_k=args.top_k,
    )

    # ------------------------------------------------------------------
    # Stage 1 + 2 output: Retrieval & Reasoning candidates
    # ------------------------------------------------------------------
    _print_section(f"Drugenix Report: {args.disease_query}")
    print(f"  Retrieved abstracts : {len(result['retrieved'])}")
    print(f"  Candidate drugs     : {len(result['candidates'])}")

    print("\n--- Reasoning Candidates (pre-validation) ---")
    for i, candidate in enumerate(result["candidates"], start=1):
        name = candidate.get("drug", "Unknown")
        confidence = candidate.get("confidence", 0.0)
        mechanism = candidate.get("mechanism", "")
        pmids = ", ".join(candidate.get("evidence_pmids", []))
        print(f"  {i}. {name}  (confidence={confidence:.2f})")
        if mechanism:
            mech_display = mechanism[:120] + "..." if len(mechanism) > 120 else mechanism
            print(f"     mechanism: {mech_display}")
        if pmids:
            print(f"     evidence PMIDs: {pmids}")

    # ------------------------------------------------------------------
    # Stage 3 output: Validation Agent results
    # ------------------------------------------------------------------
    if not args.no_validation and result.get("validation"):
        val = result["validation"]
        stats = val.get("stats", {})
        val_results = val.get("results", [])

        _print_section("Validation Agent Report", char="-")
        print(f"  Total validated : {stats.get('total', 0)}")
        print(f"  [PASS]          : {stats.get('passed', 0)}")
        print(f"  [WARN]          : {stats.get('warned', 0)}")
        print(f"  [FAIL]          : {stats.get('failed', 0)}")
        print(f"  Pass rate       : {stats.get('pass_rate', 0.0):.0%}")
        print(f"  Avg confidence  : {stats.get('avg_original_confidence', 0.0):.2f}"
              f" -> {stats.get('avg_adjusted_confidence', 0.0):.2f} (after validation)")

        top_flags = stats.get("top_safety_flags", [])
        if top_flags:
            print(f"  Safety flags    : {', '.join(top_flags)}")

        print("\n--- Validated Candidates ---")
        for r in val_results:
            # Strip emoji for Windows cp1252 terminal compatibility
            summary = r['summary'].replace("\u2705", "[PASS]").replace("\u26a0\ufe0f", "[WARN]").replace("\u274c", "[FAIL]")
            print(f"\n  {summary}")
            if r["warnings"]:
                for w in r["warnings"]:
                    print(f"    * {w}")

        passed = [r for r in val_results if r["status"] == "PASS"]
        if passed:
            _print_section("Recommended Drugs (PASS only)", char="-")
            for i, r in enumerate(passed, start=1):
                print(
                    f"  {i}. {r['drug']}  "
                    f"(adjusted confidence={r['adjusted_confidence']:.2f})"
                )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  Saved detailed JSON to: {output_path}")

    if not result["candidates"]:
        print(
            "\nNo candidates generated. Try increasing --retmax / --top-k or using a broader query.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
