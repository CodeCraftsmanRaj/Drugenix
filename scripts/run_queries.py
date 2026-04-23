from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drugenix.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Drugenix for multiple disease queries.")
    parser.add_argument("--queries", required=True, help="Text file with one disease query per line.")
    parser.add_argument("--out", default="outputs/preds.jsonl", help="Output JSONL path.")
    parser.add_argument("--retmax", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

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
