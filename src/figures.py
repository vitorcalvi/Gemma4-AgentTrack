"""Publication figures for both writeups. Regenerates every figure from results/*.json."""
from __future__ import annotations
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
INK = "#1a1a1a"
ACC = ["#4285F4", "#EA4335", "#FBBC04", "#34A853", "#9334E6"]


def fig_coverage(res="results"):
    s = json.load(open(os.path.join(res, "coverage.json")))["summary"]
    B = [1, 3, 5, 10, 20, 30, 50, 100, 200, 400, 800]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    for j, h in enumerate(range(5)):
        y = [s[f"exp{h}_cov@{b}"] for b in B]
        ax.plot(B, y, marker="o", ms=4, lw=2, color=ACC[j], label=f"{h} hop{'s' if h>1 else ''}",
                zorder=5 - h)
    ax.set_xscale("log")
    ax.set_xlabel("review budget B  (candidates the agent inspects)", fontsize=11)
    ax.set_ylabel("coverage@B  (target inside budget)", fontsize=11)
    ax.set_title("Call-graph expansion converts recall into a reviewable budget\n"
                 "SWE-bench Lite, n=%d, 10 lexical seeds" % s["n"], fontsize=12, loc="left")
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=.25, lw=.6)
    ax.legend(frameon=False, fontsize=10, title="expansion depth", title_fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_coverage.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_coverage.png"


def fig_lexical(res="results"):
    rows = [json.loads(l) for l in open(os.path.join(res, "graphloc300.jsonl"))]
    r = np.array([x["best_lexical_rank"] for x in rows], dtype=float)
    n = len(r)
    r = np.minimum(r, 30000)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    ax.hist(np.log10(r + 1), bins=45, color=ACC[0], edgecolor="white", lw=.4)
    for thr, lab in [(10, "top-10"), (100, "top-100"), (1000, "top-1,000")]:
        p = (np.array([x["best_lexical_rank"] for x in rows]) > thr).mean()
        ax.axvline(np.log10(thr + 1), color=ACC[1], ls="--", lw=1.2)
        ax.text(np.log10(thr + 1) + .04, ax.get_ylim()[1] * (.9 - .12 * [10, 100, 1000].index(thr)),
                f"  {p:.0%} of tasks\n  rank worse\n  than {lab.split('-')[1]}",
                fontsize=8, color=ACC[1], va="top")
    ax.set_xlabel("log10( lexical BM25 rank of the true patch target + 1 )", fontsize=11)
    ax.set_ylabel("tasks", fontsize=11)
    ax.set_title("Lexical retrieval has a catastrophic tail\nmedian rank %d, p90 %d, n=%d"
                 % (int(np.median(r)), int(np.percentile(r, 90)), n), fontsize=12, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lexical_tail.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_lexical_tail.png"


def fig_adherence(res="results"):
    a = json.load(open(os.path.join(res, "adherence.json")))
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    labels = ["edited the\ngold function", "did not"]
    vals = [a["resolved_given_edited"]["p"] * 100, a["resolved_given_not_edited"]["p"] * 100]
    errs = [[vals[0] - a["resolved_given_edited"]["ci"][0] * 100,
             a["resolved_given_edited"]["ci"][1] * 100 - vals[0]],
            [vals[1] - a["resolved_given_not_edited"]["ci"][0] * 100,
             a["resolved_given_not_edited"]["ci"][1] * 100 - vals[1]]]
    ax.bar(labels, vals, color=[ACC[3], ACC[1]], width=.55, zorder=3)
    ax.errorbar(labels, vals, yerr=errs, fmt="none", ecolor=INK, capsize=6, lw=1.4, zorder=4)
    for i, v in enumerate(vals):
        ax.text(i, v + errs[i][1] + 3.0, f"{v:.1f}%", ha="center", fontsize=13,
                fontweight="bold")
    ax.set_ylabel("P(task resolved)", fontsize=11)
    ax.set_ylim(0, 78)
    ax.set_title("Editing the right function is worth 8.3x\n636 agent runs, exact McNemar p = 1.9e-17",
                 fontsize=12, loc="left")
    ax.grid(axis="y", alpha=.25, lw=.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_adherence.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_adherence.png"


def fig_gap(res="results"):
    a = json.load(open(os.path.join(res, "adherence.json")))["adherence"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    ax.barh(["edited it", "IGNORED it"],
            [a["p_adhered"] * 100, a["p_gap"] * 100],
            color=[ACC[3], ACC[1]], height=.5, zorder=3)
    for i, v in enumerate([a["p_adhered"] * 100, a["p_gap"] * 100]):
        ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=13, fontweight="bold")
    ax.set_xlabel("share of runs in which the gold function was in the retriever's top-15", fontsize=10)
    ax.set_xlim(0, 66)
    ax.set_title("The adherence gap: 46.6% of retrievable targets are never edited",
                 fontsize=12, loc="left")
    ax.grid(axis="x", alpha=.25, lw=.6, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_gap.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_gap.png"


def fig_negatives(res="results"):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    names = ["BM25 order\n(control)", "hop depth,\nthen BM25", "learned GBM\nreranker", "seed\nconsensus"]
    vals = [0.493, 0.514, 0.276, 0.169]
    cols = [ACC[4], ACC[0], ACC[1], ACC[1]]
    ax.bar(names, vals, color=cols, width=.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("accuracy@20", fontsize=11)
    ax.set_ylim(0, .62)
    ax.set_title("Nothing we tried beats plain lexical ordering\n"
                 "graph features decide where to look, not what to look at first",
                 fontsize=12, loc="left")
    ax.grid(axis="y", alpha=.25, lw=.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_negatives.png"), bbox_inches="tight")
    plt.close(fig)
    return "fig_negatives.png"


def fig_card(res="results"):
    """Cover card for the Writeup (requireCardImage)."""
    s = json.load(open(os.path.join(res, "coverage.json")))["summary"]
    fig = plt.figure(figsize=(10, 5.6), dpi=170)
    fig.patch.set_facecolor("#0e1116")
    ax = fig.add_axes([0.07, 0.20, 0.86, 0.50])
    ax.set_facecolor("#0e1116")
    B = [1, 3, 5, 10, 20, 30, 50, 100, 200, 400, 800]
    for j, h in enumerate(range(5)):
        y = [s[f"exp{h}_cov@{b}"] for b in B]
        ax.plot(B, y, marker="o", ms=4.5, lw=2.6, color=ACC[j],
                label=f"{h} hop{'s' if h>1 else ''}", zorder=5 - h)
    ax.set_xscale("log")
    ax.tick_params(colors="#c8d0da", labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#3a4453")
    ax.grid(alpha=.18, color="#8fa3bd", lw=.6)
    ax.set_xlabel("review budget B", color="#c8d0da", fontsize=11)
    ax.set_ylabel("coverage@B", color="#c8d0da", fontsize=11)
    ax.legend(frameon=False, fontsize=10, labelcolor="#c8d0da", ncol=5,
              loc="upper left", bbox_to_anchor=(0, 1.13), columnspacing=1.6,
              handlelength=1.6)
    fig.text(0.07, 0.93, "Where Does the Graph Help?", color="#ffffff",
             fontsize=26, fontweight="bold")
    fig.text(0.07, 0.855, "Seed expansion for function-level bug localization on repository code graphs",
             color="#9fb0c4", fontsize=12.5)
    fig.text(0.07, 0.10, "coverage@50:  0.379 lexical  ->  0.510 (1 hop)  ->  0.562 (2 hops)      |      "
                         "0 tasks lost   |   McNemar p = 7.3e-12",
             color="#34A853", fontsize=12, fontweight="bold")
    fig.text(0.07, 0.045, "Gemma 4 Developer Agent Paper Track  ·  Vitor Calvi  ·  n=290, CPU-only, fully reproducible",
             color="#6b7c92", fontsize=9.5)
    fig.savefig(os.path.join(OUT, "cover_card.png"), facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return "cover_card.png"


if __name__ == "__main__":
    res = sys.argv[1] if len(sys.argv) > 1 else "results"
    for f in (fig_coverage, fig_lexical, fig_adherence, fig_gap, fig_negatives, fig_card):
        try:
            print("  wrote", f(res))
        except Exception as e:
            print("  FAIL", f.__name__, e)
