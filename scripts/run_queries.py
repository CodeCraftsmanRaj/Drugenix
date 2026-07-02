from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drugenix.pipeline import run_pipeline


def _set_provider_env(args: argparse.Namespace) -> None:
    if not args.provider:
        return

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Drugenix for multiple disease queries.")
    parser.add_argument("--queries", required=True, help="Text file with one disease query per line.")
    parser.add_argument("--out", default="outputs/preds.jsonl", help="Output JSONL path.")
    parser.add_argument("--retmax", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--provider", default="", help="openai | gemini | ollama | custom")
    parser.add_argument("--model", default="", help="Model name for the chosen provider.")
    parser.add_argument("--base-url", default="", help="Provider base URL.")
    parser.add_argument("--api-key", default="", help="Provider API key if needed.")
    args = parser.parse_args()

    _set_provider_env(args)

    query_lines = [
        line.strip()
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for q in query_lines:
            result = run_pipeline(disease_query=q, retmax=args.retmax, top_k=args.top_k)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"done: {q} -> {len(result.get('candidates', []))} candidates")

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
