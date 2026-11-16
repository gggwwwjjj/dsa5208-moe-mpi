"""Schematic figures for the report: the model (Section 1.1) and one expert-parallel MoE block (Section 3.2).

The numbers drawn match the default configuration in src/ (H = 512, F = 1024, E = 16, k = 2, L = 2)
and the order of operations in src/parallel/moe_parallel (route, count exchange, dispatch, sort + capacity,
expert FFN, combine).  Run: python scripts/diagrams.py  ->  results/figures/model_diagram.png, ep_diagram.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
BLUE, ORANGE, GREEN, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY, INK = "#d9d9d4", "#222222"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=9, bold=False, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.1, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def model_diagram():
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 2.6), gridspec_kw={"width_ratios": [1, 2.6]})
    for ax in (a, b):
        ax.set_axis_off()

    # (a) whole network, bottom to top
    a.set_xlim(0, 4.4); a.set_ylim(0, 10)
    a.set_title("(a) Network", fontsize=9, loc="left")
    items = [("image x (784 pixels)", "white"), ("Linear 784 → H, ReLU", GREY),
             ("MoE block 1", "#d6e6f8"), ("MoE block 2", "#d6e6f8"),
             ("Linear H → 10", GREY), ("softmax, cross-entropy", "white")]
    ys = [0.2, 1.85, 3.5, 5.15, 6.8, 8.45]
    for (t, c), y in zip(items, ys):
        box(a, 0.2, y, 3.5, 1.15, t, fc=c, fs=7.5)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(a, 1.95, y0 + 1.15, 1.95, y1)
    a.annotate("", xy=(3.85, 6.3), xytext=(3.85, 3.5), arrowprops=dict(arrowstyle="-", color="#999999"))
    a.text(3.95, 4.9, "L = 2", fontsize=7, va="center", color="#555555")

    # (b) one MoE block for one token
    b.set_xlim(0, 13); b.set_ylim(0, 10)
    b.set_title("(b) One MoE block for one token h (E = 16, k = 2)", fontsize=9, loc="left")
    box(b, 0.1, 4.3, 1.6, 1.3, "token h", fs=7.5)
    box(b, 2.3, 7.7, 4.6, 2.0, "router: p = softmax(W_r (h − mean h))\nkeeps the top 2 experts, gates g", fc="#fdeccc", fs=7)
    arrow(b, 0.9, 5.6, 2.6, 7.7)
    ex_y = [8.5, 7.05, 5.6, 4.15, 2.7, 1.25]
    labels = ["expert 1", "expert 2", "expert 3", "expert 4", "…", "expert 16"]
    chosen = {1: "g₂ = 0.64", 3: "g₄ = 0.36"}
    for i, (y, t) in enumerate(zip(ex_y, labels)):
        on = i in chosen
        box(b, 7.6, y, 2.0, 1.15, t, fc="#d6e6f8" if on else "white", ec=INK if on else "#aaaaaa", fs=7.5)
        if on:
            arrow(b, 1.7, 4.95, 7.6, y + 0.57, color=BLUE, lw=1.3)
            arrow(b, 9.6, y + 0.57, 10.6, 4.95, color=BLUE, lw=1.3)
            b.text(9.75, y + 1.15 if i == 1 else y - 0.75, chosen[i], fontsize=7, color=BLUE)
    arrow(b, 6.9, 8.2, 7.55, 7.8, color=ORANGE, ls="--")
    box(b, 10.6, 4.3, 1.6, 1.3, "h + Σ g·y", fc="#e3f5ec", fs=7.5)
    arrow(b, 12.2, 4.95, 12.95, 4.95)
    b.text(12.95, 5.45, "new h", fontsize=7, ha="right")
    arrow(b, 0.9, 4.3, 0.9, 0.2, color="#777777", style="-")
    arrow(b, 0.9, 0.2, 11.4, 0.2, color="#777777", style="-")
    arrow(b, 11.4, 0.2, 11.4, 4.3, color="#777777")
    b.text(4.2, 0.55, "residual h (all that is left if an assignment is dropped)", fontsize=6.5, color="#555555", ha="center")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "model_diagram.png", dpi=220)
    plt.close(fig)


def ep_diagram():
    """Four ranks, 16 experts (4 per rank); each rank holds 8 tokens, coloured by the rank that owns their expert.
    The destinations are made uneven on purpose (rank 0 receives 14 rows, the mean is 8), as in the real runs."""
    P, W = 4, 0.25
    cols = [BLUE, ORANGE, GREEN, PINK]
    dests = [[0, 0, 1, 2, 0, 3, 0, 1], [0, 1, 0, 2, 3, 0, 1, 0], [2, 0, 1, 0, 2, 3, 0, 2], [3, 0, 2, 1, 0, 3, 0, 1]]
    recv = [sum(row.count(d) for row in dests) for d in range(P)]
    assert sum(recv) == 32 and recv == [14, 7, 6, 5]
    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    ax.set_axis_off(); ax.set_xlim(-0.9, 15.2); ax.set_ylim(-0.7, 9.3)
    x1, x4, x5 = 0.2, 6.6, 12.4
    stages = [(x1 + 1.0, "1. route local tokens,\nbucket by owner rank"), (4.4, "2-3. count exchange,\ndispatch (all-to-all)"),
              (x4 + 1.8, "4. sort by token index,\ncapacity limit, expert FFNs"), (x5 + 1.0, "5. combine (all-to-all\nback), h + Σ g·y")]
    for x, t in stages:
        ax.text(x, 8.9, t, ha="center", va="top", fontsize=8)
    rows = [6.3, 4.6, 2.9, 1.2]
    for r, y in enumerate(rows):
        ax.text(x1 - 0.1, y + 0.55, f"rank {r}", ha="right", va="center", fontsize=8.5)
        for j, d in enumerate(dests[r]):
            ax.add_patch(FancyBboxPatch((x1 + j * W, y + 0.25), W * 0.8, 0.6, boxstyle="square,pad=0",
                                        fc=cols[d], ec="white", lw=0.5))
        for j in range(recv[r]):
            ax.add_patch(FancyBboxPatch((x4 + 0.05 + j * W, y + 0.25), W * 0.8, 0.6, boxstyle="square,pad=0",
                                        fc=cols[r], ec="white", lw=0.5))
        box(ax, x4, y - 0.02, 3.6, 1.14, "", fc="none", ec="#888888")
        ax.text(x4 + 1.8, y - 0.3, f"experts {4 * r + 1}-{4 * r + 4}: {recv[r]} rows", ha="center", fontsize=7.5,
                color=ORANGE if r == 0 else "#444444", fontweight="bold" if r == 0 else "normal")
        for j in range(8):
            ax.add_patch(FancyBboxPatch((x5 + j * W, y + 0.25), W * 0.8, 0.6, boxstyle="square,pad=0",
                                        fc="#e3f5ec", ec="#7fbf9f", lw=0.6))
    for s_, ys in enumerate(rows):
        for d, yd in enumerate(rows):
            n = dests[s_].count(d)
            ax.add_patch(FancyArrowPatch((2.3, ys + 0.55), (x4 - 0.1, yd + 0.55), arrowstyle="-|>", mutation_scale=7,
                                         color=cols[d], lw=0.5 + 0.45 * n, alpha=0.75))
            ax.add_patch(FancyArrowPatch((x4 + 3.7, yd + 0.55), (x5 - 0.1, ys + 0.55), arrowstyle="-|>", mutation_scale=6,
                                         color="#9a9a9a", lw=0.4 + 0.3 * n, alpha=0.6))
    ax.text(4.4, -0.45, "line width = number of rows sent", ha="center", fontsize=7.5, color="#444444")
    ax.text(x4 + 1.8, -0.45, "the other ranks wait until rank 0 has finished", ha="center", fontsize=7.5, color=ORANGE)
    fig.tight_layout()
    fig.savefig(OUT / "ep_diagram.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model_diagram()
    ep_diagram()
    print("wrote", OUT / "model_diagram.png", OUT / "ep_diagram.png")
