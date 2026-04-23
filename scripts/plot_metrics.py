from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required for plotting. Install with: uv add matplotlib") from exc
    return plt


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _setup_style(plt):
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "font.size": 11,
        }
    )


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=180, facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark metrics for Drugenix MVP.")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON.")
    parser.add_argument("--outdir", default="plots", help="Output directory for PNG plots.")
    args = parser.parse_args()

    plt = _require_matplotlib()
    _setup_style(plt)

    metrics = _load_json(args.metrics)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ks = [1, 3, 5, 10]

    # Precision@k curve
    p_vals = [metrics.get("retrieval", {}).get(f"P@{k}", 0.0) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, p_vals, marker="o", color="#1f77b4", linewidth=2)
    ax.set_title("Precision@k vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _save(fig, outdir / "precision_at_k.png")
    plt.close(fig)

    # nDCG@k bars
    ndcg_vals = [metrics.get("retrieval", {}).get(f"nDCG@{k}", 0.0) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar([str(k) for k in ks], ndcg_vals, color="#2ca02c")
    ax.set_title("nDCG@k")
    ax.set_xlabel("k")
    ax.set_ylabel("nDCG@k")
    ax.set_ylim(0, 1)
    for b, v in zip(bars, ndcg_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, outdir / "ndcg_at_k.png")
    plt.close(fig)

    # Top-k accuracy bars
    t_vals = [metrics.get("generation", {}).get(f"Top-{k} Accuracy", 0.0) for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar([str(k) for k in ks], t_vals, color="#ff7f0e")
    ax.set_title("Top-k Accuracy")
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    for b, v in zip(bars, t_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, outdir / "top_k_accuracy.png")
    plt.close(fig)

    # Groundedness histogram (handles missing groundedness_scores)
    grounded_scores = metrics.get("generation", {}).get("groundedness_scores", [])
    if not grounded_scores:
        mean_g = metrics.get("generation", {}).get("mean_groundedness")
        grounded_scores = [float(mean_g)] if isinstance(mean_g, (int, float)) else [0.0]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(grounded_scores, bins=max(5, min(10, len(grounded_scores))), color="#9467bd", edgecolor="#222222")
    ax.set_title("Groundedness Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, outdir / "groundedness_hist.png")
    plt.close(fig)

    # Ablation: without RAG vs with RAG
    ablation = metrics.get("ablation", {})
    labels = ["Without RAG", "With RAG"]
    correctness = [ablation.get("without_rag_correctness", 0.0), ablation.get("with_rag_correctness", 0.0)]
    grounded = [ablation.get("without_rag_groundedness", 0.0), ablation.get("with_rag_groundedness", 0.0)]

    x = list(range(2))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - width / 2 for i in x], correctness, width=width, label="Correctness", color="#17becf")
    ax.bar([i + width / 2 for i in x], grounded, width=width, label="Groundedness", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Ablation: Without RAG vs With RAG")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, outdir / "ablation_rag.png")
    plt.close(fig)

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
