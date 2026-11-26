"""
Regenerate paper Figures 4-8 from the EXISTING result JSONs. No model calls,
no benchmark runs — reads only:

  data/comparative_evaluation.json   (final 15-disease benchmark, run_benchmark.py)
  data/metrics_drugenix_mvp.json     (Hybrid retrieval P@k)
  data/gold_drugs.json               (gold standard, for nDCG recomputation)

Outputs (replaces the stale April-2026 plots):
  plots/fig4.png  Comparative evaluation (Hit@5 | Mean groundedness | Hallucination)
  plots/fig5.png  Hybrid Precision@k
  plots/fig6.png  Hybrid nDCG@k
  plots/fig7.png  Hybrid Hit@k
  plots/fig8.png  Mean groundedness + hallucination rate, three variants

Usage:  .venv/bin/python scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drugenix.evaluation import ndcg_at_k  # noqa: E402

COMP_JSON = PROJECT_ROOT / "data" / "comparative_evaluation.json"
MVP_JSON = PROJECT_ROOT / "data" / "metrics_drugenix_mvp.json"
GOLD_JSON = PROJECT_ROOT / "data" / "gold_drugs.json"
OUT_DIR = PROJECT_ROOT / "plots"

VARIANTS = ["pure_llm", "pure_graph", "hybrid_all_candidates"]
LABELS = ["Pure LLM", "Pure Graph", "Hybrid (All)"]
COLORS = ["#ff7f0e", "#2ca02c", "#1f77b4"]


def _style(plt) -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  saved {path}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _bar_labels(ax, bars, fmt: str = "{:.3f}") -> None:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015, fmt.format(h),
                ha="center", va="bottom", fontsize=9)


def main() -> None:
    comp = json.loads(COMP_JSON.read_text(encoding="utf-8"))
    mvp = json.loads(MVP_JSON.read_text(encoding="utf-8"))
    gold = json.loads(GOLD_JSON.read_text(encoding="utf-8"))

    res = {v: comp[v] for v in VARIANTS}
    used: dict[str, dict[str, float]] = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib required: .venv/bin/python scripts/make_paper_figures.py")

    _style(plt)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- Fig 4
    # Three panels straight from comparative_evaluation.json, exactly the
    # numbers in Table 4 of the paper.
    panels = [
        ("Hit@5", "hit_at_k", "Hit@5"),
        ("Mean Groundedness", "mean_groundedness", None),
        ("Hallucination Rate", "hallucination_rate", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (title, key, sub) in zip(axes, panels):
        if sub:
            vals = [res[v][key][sub] for v in VARIANTS]
        else:
            vals = [res[v][key] for v in VARIANTS]
        used.setdefault(title, dict(zip(LABELS, vals)))
        bars = ax.bar(LABELS, vals, color=COLORS, width=0.62)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="x", labelsize=9)
        _bar_labels(ax, bars)
    fig.suptitle("Comparative Evaluation: Pure LLM vs Pure Graph vs Hybrid Pipeline",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig4.png")

    # ---------------------------------------------------------------- Fig 5
    # Hybrid Precision@k from metrics_drugenix_mvp.json (identical to
    # hybrid_all_candidates.precision_at_k in the comparative file).
    ks = [1, 3, 5, 10]
    p_vals = [mvp["retrieval"][f"P@{k}"] for k in ks]
    used["Precision@k"] = dict(zip([f"P@{k}" for k in ks], p_vals))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(ks, p_vals, marker="o", color="#1f77b4", linewidth=2, markersize=7)
    for x, y in zip(ks, p_vals):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    ax.set_title("Hybrid Precision@k (15-disease benchmark)", fontsize=12)
    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig5.png")

    # ---------------------------------------------------------------- Fig 6
    # Hybrid nDCG@k recomputed from raw_predictions + gold set with the same
    # ndcg_at_k used in evaluation (matches Sec. 3.3 of the paper).
    ndcg = {k: [] for k in ks}
    for disease, preds in comp["raw_predictions"]["hybrid"].items():
        g = gold[disease]
        gs = {x.lower() for x in g["approved"] + g["repurposed"]}
        rel = [1 if p["drug"].lower() in gs else 0 for p in preds]
        for k in ks:
            ndcg[k].append(ndcg_at_k(rel, k))
    ndcg_vals = [round(sum(ndcg[k]) / len(ndcg[k]), 3) for k in ks]
    used["nDCG@k"] = dict(zip([f"nDCG@{k}" for k in ks], ndcg_vals))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar([str(k) for k in ks], ndcg_vals, color="#2ca02c", width=0.55)
    ax.set_title("Hybrid nDCG@k (15-disease benchmark)", fontsize=12)
    ax.set_xlabel("k")
    ax.set_ylabel("nDCG@k")
    ax.set_ylim(0, 1.0)
    _bar_labels(ax, bars)
    fig.tight_layout()
    _save(fig, "fig6.png")

    # ---------------------------------------------------------------- Fig 7
    # Hybrid Hit@k — title matches the caption ("Hit@k", not "Top-k Accuracy").
    hit_vals = [res["hybrid_all_candidates"]["hit_at_k"][f"Hit@{k}"] for k in ks]
    used["Hit@k"] = dict(zip([f"Hit@{k}" for k in ks], hit_vals))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar([str(k) for k in ks], hit_vals, color="#ff7f0e", width=0.55)
    ax.set_title("Hybrid Hit@k (15-disease benchmark)", fontsize=12)
    ax.set_xlabel("k")
    ax.set_ylabel("Hit@k")
    ax.set_ylim(0, 1.0)
    _bar_labels(ax, bars)
    fig.tight_layout()
    _save(fig, "fig7.png")

    # ---------------------------------------------------------------- Fig 8
    # Grouped bars: mean groundedness + hallucination rate for the three
    # variants (what the caption promises) instead of the old single
    # histogram of one score.
    g_vals = [res[v]["mean_groundedness"] for v in VARIANTS]
    h_vals = [res[v]["hallucination_rate"] for v in VARIANTS]
    used["Fig8 groundedness"] = dict(zip(LABELS, g_vals))
    used["Fig8 hallucination"] = dict(zip(LABELS, h_vals))
    import numpy as np
    x = np.arange(len(VARIANTS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x - width / 2, g_vals, width, label="Mean groundedness", color="#1f77b4")
    b2 = ax.bar(x + width / 2, h_vals, width, label="Hallucination rate", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Groundedness and hallucination by pipeline variant", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    fig.tight_layout()
    _save(fig, "fig8.png")

    # ---------------------------------------------------------- verification
    print("\nValues plotted (all traceable to the existing JSONs):")
    for section, vals in used.items():
        print(f"  {section}: " + ", ".join(f"{k}={v}" for k, v in vals.items()))


if __name__ == "__main__":
    main()
