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


def _slugify(text: str) -> str:
    out = []
    for ch in text.lower().strip():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", "/"}:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "query"


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


def _render_report(result: dict) -> str:
    query = result.get("query", "")
    runtime = result.get("runtime", {})
    retrieved = result.get("retrieved", [])
    candidates = result.get("candidates", [])

    lines: list[str] = []
    lines.append(f"# Drugenix inference report")
    lines.append("")
    lines.append(f"**Query:** {query}")
    lines.append(f"**Provider:** {runtime.get('llm_provider', '')}")
    model = runtime.get("ollama_model") or runtime.get("openai_model") or runtime.get("gemini_model") or runtime.get("custom_model")
    if model:
        lines.append(f"**Model:** {model}")
    lines.append(f"**Retrieved abstracts:** {len(retrieved)}")
    lines.append(f"**Candidate drugs:** {len(candidates)}")
    lines.append("")

    lines.append("## Candidate drugs")
    lines.append("")
    if candidates:
        lines.append("| Rank | Drug | Confidence | Evidence PMIDs | Mechanism |")
        lines.append("|---|---|---:|---|---|")
        for idx, c in enumerate(candidates, start=1):
            pmids = ", ".join(c.get("evidence_pmids", []))
            mechanism = c.get("mechanism", "").replace("\n", " ").strip()
            if len(mechanism) > 160:
                mechanism = mechanism[:157] + "..."
            lines.append(
                f"| {idx} | {c.get('drug', '')} | {float(c.get('confidence', 0.0)):.2f} | {pmids} | {mechanism} |"
            )
    else:
        lines.append("No candidates were generated.")
    lines.append("")

    lines.append("## Top retrieved evidence")
    lines.append("")
    if retrieved:
        lines.append("| PMID | Score | Title | Journal | Year |")
        lines.append("|---|---:|---|---|---|")
        for doc in retrieved[:10]:
            title = doc.get("title", "").replace("\n", " ").strip()
            if len(title) > 120:
                title = title[:117] + "..."
            lines.append(
                f"| {doc.get('pmid', '')} | {float(doc.get('retrieval_score', 0.0)):.2f} | {title} | {doc.get('journal', '')} | {doc.get('year', '')} |"
            )
    else:
        lines.append("No evidence retrieved.")

    lines.append("")
    lines.append("## Raw JSON")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result, indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Drugenix inference and save a human-readable report.")
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Disease query to infer on. If omitted, you will be prompted interactively.",
    )
    parser.add_argument("--provider", default="ollama", help="openai | gemini | ollama | custom")
    parser.add_argument("--model", default="llama3.1:8b", help="Model name.")
    parser.add_argument("--base-url", default="", help="Provider base URL.")
    parser.add_argument("--api-key", default="", help="Provider API key if needed.")
    parser.add_argument("--retmax", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--outdir", default="outputs/inference", help="Directory to write report files.")
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        query = input("Enter your disease query in natural language: ").strip()
    if not query:
        raise SystemExit("No query provided.")

    _set_provider_env(args)

    result = run_pipeline(disease_query=query, retmax=args.retmax, top_k=args.top_k)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(query)

    json_path = outdir / f"{slug}.json"
    md_path = outdir / f"{slug}.md"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_report(result), encoding="utf-8")

    print(f"Saved JSON report: {json_path}")
    print(f"Saved markdown report: {md_path}")
    print(f"Candidates: {len(result.get('candidates', []))}")
    print(f"Retrieved abstracts: {len(result.get('retrieved', []))}")

    for i, candidate in enumerate(result.get("candidates", [])[:5], start=1):
        print(f"{i}. {candidate.get('drug', '')} ({float(candidate.get('confidence', 0.0)):.2f})")


if __name__ == "__main__":
    main()
