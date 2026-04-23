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
        description="Drugenix MVP: PubMed retrieval + LLM/heuristic reasoning for drug repurposing.",
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
        help="LLM provider: openai | gemini | ollama | custom(openai-compatible).",
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
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

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

    result = run_pipeline(
        disease_query=args.disease_query,
        retmax=args.retmax,
        top_k=args.top_k,
    )

    print(f"\n=== Drugenix MVP Report: {args.disease_query} ===")
    print(f"Retrieved abstracts: {len(result['retrieved'])}")
    print(f"Candidate drugs: {len(result['candidates'])}\n")

    for i, candidate in enumerate(result["candidates"], start=1):
        name = candidate.get("drug", "Unknown")
        confidence = candidate.get("confidence", 0.0)
        mechanism = candidate.get("mechanism", "")
        pmids = ", ".join(candidate.get("evidence_pmids", []))
        print(f"{i}. {name}  (confidence={confidence:.2f})")
        if mechanism:
            print(f"   mechanism: {mechanism}")
        if pmids:
            print(f"   evidence PMIDs: {pmids}")
        print()

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved detailed JSON to: {output_path}")

    if not result["candidates"]:
        print(
            "No candidate was generated. Try increasing --retmax/--top-k or using a broader disease query.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
