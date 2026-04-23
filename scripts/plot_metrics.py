from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required for plotting. Install with: pip install matplotlib") from exc
    return plt


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark metrics for Drugenix MVP.")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON.")
    parser.add_argument("--outdir", default="plots", help="Output directory for PNG plots.")
    args = parser.parse_args()

    plt = _require_matplotlib()
    metrics = _load_json(args.metrics)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Precision@k curve
    ks = [1, 3, 5, 10]
    p_vals = [metrics.get("retrieval", {}).get(f"P@{k}", 0.0) for k in ks]
    plt.figure()
    plt.plot(ks, p_vals, marker="o")
    plt.title("Precision@k vs k")
    plt.xlabel("k")
    plt.ylabel("Precision@k")
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "precision_at_k.png", bbox_inches="tight", dpi=160)
    plt.close()

    # nDCG@k bars
    ndcg_vals = [metrics.get("retrieval", {}).get(f"nDCG@{k}", 0.0) for k in ks]
    plt.figure()
    plt.bar([str(k) for k in ks], ndcg_vals)
    plt.title("nDCG@k")
    plt.xlabel("k")
    plt.ylabel("nDCG@k")
    plt.savefig(outdir / "ndcg_at_k.png", bbox_inches="tight", dpi=160)
    plt.close()

    # Top-k accuracy bars
    t_vals = [metrics.get("generation", {}).get(f"Top-{k} Accuracy", 0.0) for k in ks]
    plt.figure()
    plt.bar([str(k) for k in ks], t_vals)
    plt.title("Top-k Accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.savefig(outdir / "top_k_accuracy.png", bbox_inches="tight", dpi=160)
    plt.close()

    # Groundedness histogram
    grounded_scores = metrics.get("generation", {}).get("groundedness_scores", [])
    plt.figure()
    plt.hist(grounded_scores, bins=10)
    plt.title("Groundedness Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.savefig(outdir / "groundedness_hist.png", bbox_inches="tight", dpi=160)
    plt.close()

    # Ablation: without RAG vs with RAG
    ablation = metrics.get("ablation", {})
    labels = ["Without RAG", "With RAG"]
    correctness = [ablation.get("without_rag_correctness", 0.0), ablation.get("with_rag_correctness", 0.0)]
    grounded = [ablation.get("without_rag_groundedness", 0.0), ablation.get("with_rag_groundedness", 0.0)]

    x = range(2)
    width = 0.35
    plt.figure()
    plt.bar([i - width / 2 for i in x], correctness, width=width, label="Correctness")
    plt.bar([i + width / 2 for i in x], grounded, width=width, label="Groundedness")
    plt.xticks(list(x), labels)
    plt.ylim(0, 1)
    plt.title("Ablation: Without RAG vs With RAG")
    plt.legend()
    plt.savefig(outdir / "ablation_rag.png", bbox_inches="tight", dpi=160)
    plt.close()

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
