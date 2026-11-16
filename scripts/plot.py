"""Draw every report figure from results/tables/*.csv.

Rules from L12 p36: axes for time, problem size and core count are
logarithmic; speedup and efficiency are shown side by side with the ideal line.
One function per figure; figures for experiments without a CSV are skipped.
"""
import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Categorical slots in fixed order (validated: scripts in the dataviz guide, light
# surface; three slots are below 3:1 contrast, so every series also has its own
# marker and the report gives the numbers as tables).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
# phases of a training step, in the order they happen (train.cpp)
PHASES = [("t_compute", "compute"), ("t_a2a_wait", "all-to-all wait (imbalance)"), ("t_a2a", "all-to-all transfer"),
          ("t_wait", "all-reduce wait"), ("t_comm", "all-reduce / ZeRO comm"), ("t_optim", "optimiser")]


def style(ax):
    ax.grid(True, which="major", color="#e4e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def cores(r):
    return int(r["ranks"]) * int(r["threads"])


def read(path: pathlib.Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def scaling_figure(cores, times, t_serial, title, out, weak=False):
    """Speedup (strong) or time (weak) next to efficiency; log2 core axis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    if weak:
        ax1.plot(cores, times, "o-")
        ax1.set_ylabel("time per step (s)")
        efficiency = [times[0] / t for t in times]
    else:
        speedup = [t_serial / t for t in times]
        ax1.plot(cores, speedup, "o-", label="measured")
        ax1.plot(cores, cores, "k--", label="ideal")
        ax1.set_ylabel("speedup")
        ax1.legend()
        efficiency = [s / p for s, p in zip(speedup, cores)]
    ax2.plot(cores, efficiency, "o-")
    ax2.axhline(1.0, color="k", ls="--")
    ax2.set_ylabel("parallel efficiency")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("CPU cores")
    if not weak:
        ax1.set_yscale("log", base=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_pingpong(tables: pathlib.Path, out: pathlib.Path) -> None:
    """E0: one-way time vs message size, same node vs two nodes, per cluster."""
    path = tables / "E0.csv"
    if not path.exists():
        return
    rows = [r for r in read(path) if r.get("kind") == "pingpong" and r.get("platform") in ("atlas", "vanda")]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    i = 0
    for plat in ("vanda", "atlas"):
        for link in ("intra", "inter"):
            sel = sorted((r for r in rows if r["platform"] == plat and r["log"] == f"pingpong_{link}.log"),
                         key=lambda r: int(r["bytes"]))
            if not sel:
                continue
            ax.loglog([int(r["bytes"]) for r in sel], [float(r["t_oneway"]) for r in sel], marker=MARKERS[i],
                      color=SERIES[i], lw=1.5, ms=4, label=f"{plat.capitalize()}, {'same node' if link == 'intra' else 'two nodes'}")
            i += 1
    ax.set_xscale("log", base=2)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("one-way time (s)")
    ax.legend(fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "e0_pingpong.png", dpi=200)
    plt.close(fig)


def median_runs(rows, prefix, key):
    """E3-style repeated runs: logs <prefix><P>_<key>.log and <prefix><P>_<key>_r<n>.log.
    Returns one row per rank count - the run with the median step time (lower
    median for an even count) - with 'n_reps', 't_min', 't_max' added."""
    import re

    pat = re.compile(rf"^{re.escape(prefix)}(\d+)_{key}(_r\d+)?\.log$")
    groups = {}
    for r in rows:
        m = pat.match(r["log"])
        if m and int(r.get("threads") or 1) == 1:
            groups.setdefault(int(m.group(1)), []).append(r)
    out = []
    for p in sorted(groups):
        runs = sorted(groups[p], key=lambda r: float(r["t_step_mean"]))
        med = dict(runs[(len(runs) - 1) // 2])
        med.update(n_reps=len(runs), t_min=float(runs[0]["t_step_mean"]), t_max=float(runs[-1]["t_step_mean"]))
        out.append(med)
    return out


def fig_e3(tables: pathlib.Path, out: pathlib.Path, plat: str = "vanda") -> None:
    """E3 strong scaling: speedup and efficiency vs cores, and the share of each
    phase in the step. Speedup in the figure is relative to the parallel program
    on one rank, T_1 (same binary; for MoE the mean of the two 1-rank runs, which
    differ only by noise because nothing is communicated). The serial program is
    slower than T_1 for a compiler reason (built without -fopenmp, EXTRA), so the
    speedup against it is given in E3_<plat>_summary.csv as well."""
    path = tables / "E3.csv"
    if not path.exists():
        return
    rows = [r for r in read(path) if r.get("platform") == plat]
    serial = [r for r in rows if r["log"] == "e3_serial15.log"] or [r for r in rows if r["log"] == "e3_serial.log"]
    if not serial:
        return
    t_s = float(serial[0]["t_step_mean"])
    one = lambda key: [float(r["t_step_mean"]) for r in rows if r["log"] == f"e3_p1_{key}.log"]
    t1_moe = sum(one("mpi") + one("ring")) / max(len(one("mpi") + one("ring")), 1)
    series = []  # label, rows sorted by cores (median of repeated runs), base time T_1
    for key, label in (("mpi", "MoE, MPI_Allreduce"), ("ring", "MoE, ring all-reduce"), ("dense", "dense FFN (DP only)")):
        sel = median_runs(rows, "e3_p", key)
        if sel:
            base = t1_moe if key != "dense" else float(next((r["t_step_mean"] for r in sel if cores(r) == 1), "nan"))
            series.append((label, sel, base))
    hybrid = [r for r in rows if r["log"].startswith("e3_p32_t3")]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    summary = []
    for i, (label, sel, base) in enumerate(series):
        x = [cores(r) for r in sel]
        sp = [base / float(r["t_step_mean"]) for r in sel]
        ax1.plot(x, sp, marker=MARKERS[i], color=SERIES[i], lw=1.5, ms=4, label=label)
        # range over repeated runs (speedup of the slowest and the fastest run)
        lo = [base / r["t_max"] for r in sel]
        hi = [base / r["t_min"] for r in sel]
        ax1.errorbar(x, sp, yerr=[[v - l for v, l in zip(sp, lo)], [h - v for v, h in zip(sp, hi)]],
                     fmt="none", ecolor=SERIES[i], elinewidth=1, capsize=2)
        ax2.plot(x, [v / c for v, c in zip(sp, x)], marker=MARKERS[i], color=SERIES[i], lw=1.5, ms=4, label=label)
        for r, v in zip(sel, sp):
            summary.append({"series": label, "cores": cores(r), "ranks": r["ranks"], "threads": r["threads"],
                            "t_step_s": float(r["t_step_mean"]), "speedup": round(v, 2),
                            "efficiency": round(v / cores(r), 3),
                            "speedup_vs_serial": round(t_s / float(r["t_step_mean"]), 2) if label.startswith("MoE") else "",
                            "n_reps": r["n_reps"], "t_min_s": r["t_min"], "t_max_s": r["t_max"]})
    for r in hybrid:
        v = t1_moe / float(r["t_step_mean"])
        ax1.plot([cores(r)], [v], marker="*", color="#4a3aa7", ms=9, ls="none", label="MoE, 32 ranks x 3 threads")
        ax2.plot([cores(r)], [v / cores(r)], marker="*", color="#4a3aa7", ms=9, ls="none")
        summary.append({"series": "MoE, 32 ranks x 3 threads", "cores": cores(r), "ranks": r["ranks"],
                        "threads": r["threads"], "t_step_s": float(r["t_step_mean"]), "speedup": round(v, 2),
                        "efficiency": round(v / cores(r), 3), "speedup_vs_serial": round(t_s / float(r["t_step_mean"]), 2),
                        "n_reps": 1, "t_min_s": float(r["t_step_mean"]), "t_max_s": float(r["t_step_mean"])})
    xs = sorted({cores(r) for _, sel, _ in series for r in sel})
    if xs:
        ax1.plot(xs, xs, "k--", lw=1, label="ideal")
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("CPU cores")
        style(ax)
    ax1.set_yscale("log", base=2)
    ax1.set_ylabel(r"speedup $T_1 / T_P$")
    ax2.set_ylabel(r"parallel efficiency $T_1 / (P\,T_P)$")
    ax1.legend(fontsize=8.5)
    # phase shares of the MoE (MPI_Allreduce) step
    sel = series[0][1] if series else []
    if sel:
        labels = [str(cores(r)) for r in sel]
        bottom = [0.0] * len(sel)
        for j, (col, name) in enumerate(PHASES):
            share = [float(r.get(col) or 0) / sum(float(r.get(c) or 0) for c, _ in PHASES) for r in sel]
            ax3.bar(labels, share, bottom=bottom, color=SERIES[j], edgecolor="white", linewidth=1, label=name)
            bottom = [b + v for b, v in zip(bottom, share)]
        ax3.set_xlabel("CPU cores (MoE, MPI_Allreduce)")
        ax3.set_ylabel("share of step time")
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        style(ax3)
    fig.tight_layout()
    fig.savefig(out / f"e3_strong_{plat}.png", dpi=200)
    plt.close(fig)
    if summary:
        with open(tables / f"E3_{plat}_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)


def fig_e4(tables: pathlib.Path, out: pathlib.Path, plat: str = "vanda") -> None:
    """E4 weak scaling (512 tokens per rank): time per step and weak efficiency
    t(smallest P) / t(P) for the three layouts."""
    path = tables / "E4.csv"
    if not path.exists():
        return
    rows = [r for r in read(path) if r.get("platform") == plat]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (key, label) in enumerate((("grid", "MoE, E = 16 (DP x EP grid)"), ("eq", "MoE, E = P (one expert per rank)"),
                                      ("dense", "dense FFN (DP only)"))):
        sel = sorted((r for r in rows if r["log"].startswith(f"e4_{key}_p")), key=cores)
        if not sel:
            continue
        x = [cores(r) for r in sel]
        t = [float(r["t_step_mean"]) for r in sel]
        ax1.plot(x, t, marker=MARKERS[i], color=SERIES[i], lw=1.5, ms=4, label=label)
        ax2.plot(x, [t[0] / v for v in t], marker=MARKERS[i], color=SERIES[i], lw=1.5, ms=4, label=label)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("CPU cores (global batch = 512 x cores)")
        style(ax)
    ax1.set_yscale("log")
    ax1.set_ylabel("time per training step (s)")
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.set_ylabel("weak-scaling efficiency")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"e4_weak_{plat}.png", dpi=200)
    plt.close(fig)


def fig_e5(tables: pathlib.Path, out: pathlib.Path, plat: str = "vanda") -> None:
    """E5: all-reduce time vs message size (MPI_Allreduce vs our ring), and the
    token exchange alone (MPI_Alltoallv vs pairwise Isend/Irecv) vs ranks."""
    path = tables / "E5.csv"
    if not path.exists():
        return
    rows = [r for r in read(path) if r.get("platform") == plat]
    ar = [r for r in rows if r.get("kind") == "allreduce"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, variant, title in ((axes[0], "mpi", "MPI_Allreduce"), (axes[1], "ring", "ring all-reduce (ours)")):
        for i, p in enumerate(sorted({int(r["ranks"]) for r in ar})):
            sel = sorted((r for r in ar if r["variant"] == variant and int(r["ranks"]) == p), key=lambda r: int(r["bytes"]))
            ax.loglog([int(r["bytes"]) for r in sel], [float(r["t"]) for r in sel], marker=MARKERS[i % 6],
                      color=SERIES[i % 6], lw=1.5, ms=3, label=f"{p} ranks")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("message size (bytes)")
        ax.set_ylabel("time per all-reduce (s)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8.5)
        style(ax)
    a2a = [r for r in rows if r.get("kind", "").startswith("ep_")]
    for i, (variant, label) in enumerate((("mpi", "MPI_Alltoallv"), ("pairwise", "pairwise Isend/Irecv"))):
        sel = sorted((r for r in a2a if r["log"].startswith(f"e5_a2a_{variant}_")), key=cores)
        if sel:
            axes[2].plot([cores(r) for r in sel], [float(r["t_a2a"]) for r in sel], marker=MARKERS[i], color=SERIES[i],
                         lw=1.5, ms=4, label=label)
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("ranks (E3 problem)")
    axes[2].set_ylabel("token exchange per step, excl. waiting (s)")
    axes[2].legend(fontsize=8.5)
    style(axes[2])
    fig.tight_layout()
    fig.savefig(out / f"e5_comm_{plat}.png", dpi=200)
    plt.close(fig)


def fig_e6(tables: pathlib.Path, out: pathlib.Path, plat: str = "atlas") -> None:
    """E6 load balance (Fashion-MNIST, 16 ranks, E = 16, 5 epochs): max/mean
    expert load during training for three aux weights, the dropped share per
    aux weight and capacity factor, and test accuracy per aux weight."""
    if not (tables / "E6.csv").exists() or not (tables / "E6_load.csv").exists():
        return
    import re
    rows = [r for r in read(tables / "E6.csv") if r.get("platform") == plat]
    load = [r for r in read(tables / "E6_load.csv") if r.get("platform") == plat]
    key = lambda log: re.match(r"e6_aux([\d.]+)_cf([\d.]+)_k(\d)_s(\d+)", log).groups()
    auxes = ["0", "0.01", "0.1"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for i, aux in enumerate(auxes):
        sel = sorted((r for r in load if r["log"] == f"e6_aux{aux}_cf1.25_k2_s42.log"), key=lambda r: int(r["step"]))
        axes[0].plot([int(r["step"]) for r in sel], [float(r["imbalance"]) for r in sel], marker=MARKERS[i],
                     color=SERIES[i], lw=1.5, ms=3, label=f"aux weight {aux}")
    axes[0].axhline(8, color="k", ls=":", lw=1)
    axes[0].text(600, 7.75, "maximum E/k = 8", fontsize=7, ha="center", va="top")
    axes[0].set_xlabel("training step (B = 256)")
    axes[0].set_ylabel("max / mean tokens per expert")
    axes[0].legend(fontsize=8.5)
    for i, cf in enumerate(["1.0", "1.25", "2.0"]):
        vals = []
        for aux in auxes:
            r = next((r for r in rows if key(r["log"]) == (aux, cf, "2", "42")), None)
            vals.append(float(r["dropped"]) if r else 0.0)
        axes[1].bar([j + (i - 1) * 0.27 for j in range(3)], vals, width=0.25, color=SERIES[i], edgecolor="white",
                    label=f"capacity factor {cf}")
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels([f"aux {a}" for a in auxes])
    axes[1].set_ylabel("share of assignments dropped (k = 2)")
    axes[1].legend(fontsize=8.5)
    for i, aux in enumerate(auxes):
        acc = [float(r["test_acc"]) * 100 for r in rows if key(r["log"])[:3] == (aux, "1.25", "2")]
        axes[2].plot([i] * len(acc), acc, marker=MARKERS[i], color=SERIES[i], ls="none", ms=6)
        if acc:
            axes[2].plot([i - 0.2, i + 0.2], [sum(acc) / len(acc)] * 2, color="k", lw=1.5)
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels([f"aux {a}" for a in auxes])
    axes[2].set_xlim(-0.5, 2.5)
    axes[2].set_ylabel("test accuracy (%), cf 1.25, k = 2")
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(out / f"e6_load_{plat}.png", dpi=200)
    plt.close(fig)


def fig_e8(tables: pathlib.Path, out: pathlib.Path, plat: str = "vanda") -> None:
    """E8: time per step for each ranks x threads split of one node and of 256 cores."""
    path = tables / "E8.csv"
    if not path.exists():
        return
    rows = [r for r in read(path) if r.get("platform") == plat]
    groups = [("e8_node_", "one node"), ("e8_256_", "256 cores")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (prefix, title) in zip(axes, groups):
        import re
        groups = {}
        for r in rows:
            if r["log"].startswith(prefix):
                groups.setdefault(re.sub(r"_r\d+\.log$", ".log", r["log"]), []).append(float(r["t_step_mean"]))
        sel = sorted((next(r for r in rows if r["log"] == name) for name in groups), key=lambda r: int(r["ranks"]))
        if not sel:
            continue
        labels = [f"{r['ranks']}x{r['threads']}" for r in sel]
        runs = [sorted(groups[r["log"]]) for r in sel]
        med = [v[(len(v) - 1) // 2] for v in runs]  # median run (lower median for an even count)
        ax.bar(labels, med, color=SERIES[0], edgecolor="white")
        if any(len(v) > 1 for v in runs):
            ax.errorbar(labels, med, yerr=[[m - v[0] for m, v in zip(med, runs)], [v[-1] - m for m, v in zip(med, runs)]],
                        fmt="none", ecolor="#333333", elinewidth=1, capsize=3)
        ax.set_xlabel("MPI ranks x OpenMP threads per rank")
        ax.set_ylabel("time per training step (s)")
        ax.set_title(f"{title}, E3 problem" if title.endswith("cores") else f"{title} ({cores(sel[0])} cores), E3 problem", fontsize=9)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / f"e8_hybrid_{plat}.png", dpi=200)
    plt.close(fig)


def fig_e9(tables: pathlib.Path, out: pathlib.Path, plat: str = "vanda") -> None:
    """E9: one A40 (CUDA backend) vs one CPU core of the same node, time per step
    vs batch B and vs width H, with the GPU/CPU speedup written at each point."""
    path = tables / "E9.csv"
    if not path.exists():
        return
    rows = [r for r in read(path) if r.get("platform") == plat]
    panels = [("batch", "global batch size B (H = 512)", "_b"), ("hidden", "hidden width H (F = 2H, B = 4096)", "_h")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, (col, label, tag) in zip(axes, panels):
        curves = {}
        for i, (dev, name) in enumerate([("gpu", "1 A40 GPU"), ("cpu", "1 CPU core")]):
            sel = sorted((r for r in rows if r["log"].startswith(f"e9_{dev}{tag}")), key=lambda r: float(r[col]))
            if not sel:
                continue
            x = [float(r[col]) for r in sel]
            t = [float(r["t_step_mean"]) for r in sel]
            curves[dev] = dict(zip(x, t))
            ax.loglog(x, t, marker=MARKERS[i], color=SERIES[i], label=name)
        if "gpu" in curves and "cpu" in curves:
            for x, tg in curves["gpu"].items():
                if x in curves["cpu"]:
                    ax.annotate(f"{curves['cpu'][x] / tg:.0f}x", (x, tg), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=7)
        if col == "batch":
            # reference: the fastest one-node CPU split of E8 (same problem at B = 16384)
            e8 = tables / "E8.csv"
            node = [r for r in read(e8) if r.get("platform") == plat and r["log"] == "e8_node_r64_t1.log"] if e8.exists() else []
            if node:
                ax.loglog([16384], [float(node[0]["t_step_mean"])], marker="*", markersize=10, linestyle="none",
                          color=SERIES[2], label="1 CPU node (64 ranks, E8)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(label)
        ax.set_ylabel("time per training step (s)")
        ax.legend(fontsize=8)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / f"e9_gpu_{plat}.png", dpi=200)
    plt.close(fig)


def fig_e10(tables: pathlib.Path, out: pathlib.Path, plat: str = "vanda") -> None:
    """E10: MPI + CUDA on 1 and 2 GPUs, time per step split into phases
    (stacked bars), one panel per batch size."""
    path = tables / "E10.csv"
    if not path.exists():
        return
    rows = [r for r in read(path) if r.get("platform") == plat]
    order = [("e10_g1_", "1 GPU"), ("e10_g2_dp_", "2 GPUs, DP"), ("e10_g2_dp_zero_", "2 GPUs, DP\n+ ZeRO-1"),
             ("e10_g2_ep_", "2 GPUs, EP")]
    batches = sorted({int(float(r["batch"])) for r in rows})
    if not batches:
        return
    fig, axes = plt.subplots(1, len(batches), figsize=(4.6 * len(batches), 3.6), squeeze=False)
    for ax, b in zip(axes[0], batches):
        sel, labels = [], []
        for prefix, label in order:
            hit = [r for r in rows if r["log"].startswith(prefix) and int(float(r["batch"])) == b]
            # "e10_g2_dp_" also matches the ZeRO runs: keep exact names only
            hit = [r for r in hit if r["log"] == f"{prefix}b{b}.log"]
            if hit:
                sel.append(hit[0])
                labels.append(label)
        bottom = [0.0] * len(sel)
        for i, (col, name) in enumerate(PHASES):
            vals = [float(r[col] or 0) for r in sel]
            ax.bar(labels, vals, bottom=bottom, color=SERIES[i], edgecolor="white", label=name)
            bottom = [a + v for a, v in zip(bottom, vals)]
        ax.set_ylabel("time per training step (s)")
        ax.set_title(f"B = {b}", fontsize=9)
        style(ax)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / f"e10_multigpu_{plat}.png", dpi=200)
    plt.close(fig)


def e1_params(r) -> float:
    """Trainable parameters of the logged model (dense part + all experts)."""
    H, F, E, L = (int(float(r[c])) for c in ("hidden", "ffn", "experts", "blocks"))
    dense = 784 * H + H + 10 * H + 10
    if E <= 1:  # dense baseline: L residual FFN blocks
        return dense + L * (2 * H * F + H + F)
    return dense + L * (H * E + E) + L * E * (2 * H * F + H + F)


def fig_e1(tables: pathlib.Path, out: pathlib.Path) -> None:
    """Serial step time vs batch, width, experts, k and depth (grading item 2).

    A step has two parts with different scaling: forward + backward (compute,
    proportional to the flop count) and the Adam update (proportional to the
    number of parameters, all E experts included). Each is drawn with its own
    model line: flops / rate and parameters x time per parameter, the rate and the
    time per parameter taken as medians over all E1 runs."""
    import numpy as np

    path = tables / "E1.csv"
    if not path.exists():
        return
    # Vanda runs only (same machine as E3); laptop development logs are never plotted
    rows = [r for r in read(path) if r.get("kind") == "serial" and r.get("platform") == "vanda"
            and r["log"].startswith("e1_") and not r["log"].startswith("e1_dense")]
    if not rows:
        return
    flops = lambda r: float(r["gflops"]) * 1e9 * float(r["t_step_mean"])  # nominal flops per step
    rate = np.median([flops(r) / float(r["t_compute"]) for r in rows if r["log"].startswith("e1_batch")
                      and float(r["batch"]) >= 1024])
    per_param = np.median([float(r["t_optim"]) / e1_params(r) for r in rows])
    panels = [("batch", "global batch size B", "e1_batch", "Batch size B: compute grows, Adam stays flat"),
              ("hidden", "hidden width H (F = 2H)", "e1_hidden", "Width H: both parts grow as about H\u00b2"),
              ("experts", "experts per block E (k = 2)", "e1_experts", "Experts E: Adam grows with E, compute little"),
              ("topk", "experts per token k (E = 16)", "e1_topk", "Experts per token k: compute grows, Adam flat"),
              ("blocks", "number of MoE blocks L", "e1_blocks", "MoE blocks L: both parts grow with L")]
    fig, grid = plt.subplots(2, 3, figsize=(10.5, 6.6))
    axes = grid.ravel()
    for ax, (col, label, prefix, title) in zip(axes, panels):
        sel = sorted((r for r in rows if r["log"].startswith(prefix)), key=lambda r: float(r[col]))
        if not sel:
            continue
        x = np.array([float(r[col]) for r in sel])
        ax.loglog(x, [float(r["t_step_mean"]) for r in sel], color="#9a9a9a", lw=1.2, marker="o", ms=3,
                  label="whole step (measured)")
        ax.loglog(x, [float(r["t_compute"]) for r in sel], marker="s", color=SERIES[1], lw=1.6,
                  label="forward + backward pass (measured)")
        ax.loglog(x, [flops(r) / rate for r in sel], "--", color=SERIES[1], linewidth=1.1,
                  label=f"model: operation count / {rate / 1e9:.1f} GFLOP/s")
        ax.loglog(x, [float(r["t_optim"]) for r in sel], marker="^", color=SERIES[2], lw=1.6,
                  label="Adam update (measured)")
        ax.loglog(x, [e1_params(r) * per_param for r in sel], ":", color="#0b5d3f", linewidth=1.6,
                  label=f"model: parameter count \u00d7 {per_param * 1e9:.1f} ns")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(label)
        ax.set_ylabel("time per training step (s)")
        ax.set_title(title, fontsize=9, loc="left")
        style(ax)
    guide = axes[-1]
    guide.axis("off")
    handles, names = axes[0].get_legend_handles_labels()
    guide.legend(handles, names, loc="upper left", fontsize=8.5, frameon=False)
    guide.text(0.0, 0.42, "How to read: each panel varies one size parameter,\n"
               "the others stay at the default (B = 256, H = 512,\nE = 16, k = 2, L = 2). Both axes are logarithmic.\n"
               "Where a measured line follows its model line,\nthat part scales as predicted: compute with the\n"
               "number of operations, Adam with the number of\nparameters (all E experts, used or not).",
               fontsize=8.5, va="top", transform=guide.transAxes, color="#333333")
    fig.tight_layout()
    fig.savefig(out / "e1_serial_scaling.png", dpi=200)
    plt.close(fig)


def fig_gemm(tables: pathlib.Path, out: pathlib.Path) -> None:
    path = tables / "E0.csv"
    if not path.exists():
        return
    # the E1 job's benchmark (baseline build); EXTRA's -march=native comparison is a table in the report
    rows = [r for r in read(path) if r.get("kind") == "gemm" and r.get("platform") == "vanda" and r["log"] == "e1_gemm.log"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for impl in ["naive", "blocked"]:
        for threads in sorted({r["threads"] for r in rows}, key=int):
            sel = sorted((r for r in rows if r["impl"] == impl and r["threads"] == threads), key=lambda r: int(r["n"]))
            if sel:
                ax.loglog([int(r["n"]) for r in sel], [float(r["gflops"]) for r in sel], "o-",
                          label=f"{impl}, {threads} thread(s)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("GFLOP/s")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "e0_gemm.png", dpi=200)
    plt.close(fig)


CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def fig_e11(tables: pathlib.Path, out: pathlib.Path) -> None:
    """Routing heatmap (E11): share of each Fashion-MNIST class's top-k
    assignments that go to each expert, on the test set after training. Taken
    from the E6 runs with and without the aux loss (cf 1.25, k 2, seed 42)."""
    path = tables / "E6_route.csv"
    if not path.exists():
        return
    rows = read(path)
    runs = [("e6_aux0.1_cf1.25_k2_s42.log", "aux 0.1"), ("e6_aux0_cf1.25_k2_s42.log", "aux 0")]
    runs = [(log, label) for log, label in runs if any(r["log"] == log for r in rows)]
    if not runs:
        return
    layers = sorted({int(r["layer"]) for r in rows})
    fig, axes = plt.subplots(len(runs), len(layers), figsize=(4.2 * len(layers), 3.2 * len(runs)), squeeze=False)
    for i, (log, label) in enumerate(runs):
        for j, layer in enumerate(layers):
            sel = sorted((r for r in rows if r["log"] == log and int(r["layer"]) == layer), key=lambda r: int(r["class"]))
            counts = [[float(v) for v in r["counts"].split(":")] for r in sel]
            share = [[v / sum(row) for v in row] for row in counts]
            ax = axes[i][j]
            im = ax.imshow(share, aspect="auto", cmap="viridis", vmin=0, vmax=max(max(r) for r in share))
            ax.set_yticks(range(len(sel)))
            ax.set_yticklabels([CLASSES[int(r["class"])] for r in sel], fontsize=7)
            ax.set_xlabel("expert")
            ax.set_title(f"{label}, MoE layer {layer + 1}", fontsize=9)
            fig.colorbar(im, ax=ax, label="share of class")
    fig.tight_layout()
    fig.savefig(out / "e11_routing.png", dpi=200)
    plt.close(fig)


def fig_e11_confusion(tables: pathlib.Path, out: pathlib.Path) -> None:
    """E11: test-set confusion matrix (row = true class, share of the row) and
    per-class accuracy of the full training run."""
    path = tables / "E11_conf.csv"
    if not path.exists():
        return
    rows = read(path)
    log = rows[0]["log"]
    sel = sorted((r for r in rows if r["log"] == log), key=lambda r: int(r["class"]))
    counts = [[float(v) for v in r["counts"].split(":")] for r in sel]
    share = [[v / sum(row) for v in row] for row in counts]
    names = [CLASSES[int(r["class"])] for r in sel]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    im = ax1.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel("predicted class")
    ax1.set_ylabel("true class")
    fig.colorbar(im, ax=ax1, label="share of true class")
    ax2.barh(range(len(names)), [share[i][i] for i in range(len(names))], color=SERIES[0])
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("per-class test accuracy")
    fig.tight_layout()
    fig.savefig(out / "e11_confusion.png", dpi=200)
    plt.close(fig)


def fig_e11_similarity(tables: pathlib.Path, out: pathlib.Path) -> None:
    """E11: for every pair of classes, how similarly the trained router sends them
    to experts (Bhattacharyya coefficient of the two classes' expert distributions
    on the test set) against how often the model confuses them (both directions)."""
    import numpy as np

    route, conf = tables / "E11_route.csv", tables / "E11_conf.csv"
    if not route.exists() or not conf.exists():
        return
    counts = lambda rows: np.array([[int(v) for v in r["counts"].split(":")]
                                    for r in sorted(rows, key=lambda r: int(r["class"]))], float)
    m = counts(read(conf))
    confused = m + m.T
    iu = np.triu_indices(len(CLASSES), 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for layer, ax in enumerate(axes):
        r = counts([x for x in read(route) if int(x["layer"]) == layer])
        r /= r.sum(axis=1, keepdims=True)
        sim = np.sqrt(r) @ np.sqrt(r).T
        x, y = sim[iu], confused[iu]
        from scipy.stats import spearmanr
        rho = spearmanr(x, y)[0]
        ax.scatter(x, y + 1, color=SERIES[0], s=18)
        # number the five most confused pairs next to their points and name them in a key (no crossing labels)
        top = sorted(np.argsort(-y)[:5], key=lambda k: -(y[k]))
        key = []
        for n, k in enumerate(top, 1):
            ax.scatter([x[k]], [y[k] + 1], s=60, facecolors="none", edgecolors="#333333", linewidths=0.8)
            ax.annotate(str(n), (x[k], y[k] + 1), xytext=(-7, 5), textcoords="offset points", fontsize=7,
                        ha="right", fontweight="bold")
            key.append(f"{n}  {CLASSES[iu[0][k]]}\u2013{CLASSES[iu[1][k]]} ({int(y[k])})")
        ax.text(0.02, 0.97, "most confused pairs:\n" + "\n".join(key), transform=ax.transAxes, fontsize=6.5,
                va="top", ha="left", color="#333333")
        ax.set_yscale("log")
        ax.set_ylim(0.7, (y.max() + 1) * 3)
        ax.set_xlabel("routing similarity of the two classes")
        ax.set_title(f"MoE layer {layer + 1} (Spearman \u03c1 = {rho:.2f}, 45 pairs)", fontsize=9)
        style(ax)
    axes[0].set_ylabel("test images confused (both ways) + 1")
    fig.tight_layout()
    fig.savefig(out / "e11_similarity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", default="results/tables")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    tables, out = pathlib.Path(args.tables), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_pingpong(tables, out)
    fig_gemm(tables, out)
    fig_e1(tables, out)
    fig_e3(tables, out)
    fig_e4(tables, out)
    fig_e5(tables, out)
    fig_e6(tables, out)
    fig_e8(tables, out)
    fig_e9(tables, out)
    fig_e10(tables, out)
    fig_e11(tables, out)
    fig_e11_confusion(tables, out)
    fig_e11_similarity(tables, out)
    # TODO: E9/E10 GPU, cost model overlay.


if __name__ == "__main__":
    main()
