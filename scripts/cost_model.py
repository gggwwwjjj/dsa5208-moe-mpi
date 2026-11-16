"""Cost model for one training step (plan §5.6, L12 p21) and comparison with E3.

    python scripts/cost_model.py [--platform vanda]

    T_step(P) = T_compute + T_alltoall + T_allreduce + T_optimiser

    T_compute   = C_dense / P + C_expert * r(P) / P
                  C_dense, C_expert: the 1-rank compute time (parallel program,
                  same binary as all runs) split by flop count. r(P): the busiest
                  rank sets the pace. The RESULT line gives max/mean tokens per
                  *expert* before the capacity cut; an expert keeps at most cf
                  times the mean, so r_expert = min(imbalance, cf). With one expert
                  per rank (ep = E) that is the rank imbalance, with all experts on
                  one rank (ep = 1) there is none, and in between we interpolate
                  linearly in ep/E. Compared with the measured compute + all-to-all
                  wait (the busiest rank). Ideal gamma: no slowdown when a node is
                  full and no loss for small matrices - the measured deviations
                  from this term are discussed in the report (Section 6).
    T_alltoall  = 4 L [ (ep-1) alpha_intra + (B/P) k (H+2) s (ep-1)/ep / beta_intra ]
                  (every EP group of <= 16 ranks lies inside one node)
    T_allreduce = ring model, dense gradients over all P ranks, then expert
                  gradients over the D = P/ep copies of each expert:
                  2(Q-1) alpha + 2 (Q-1)/Q n s / beta_eff for a group of Q ranks.
                  Isolated-pair model: beta_eff = beta of one ping-pong pair.
                  Shared-link model: the 16 expert-DP groups all-reduce at the
                  same time and each crosses every node boundary, so they share
                  the node's network bandwidth: beta_eff = min(beta, B_node / 16),
                  B_node = total bandwidth of 64 concurrent pairs between two
                  nodes (E0 --concurrent).
    T_optimiser = parameters held by the rank x Adam time per parameter (1 rank).

alpha, beta from the E0 ping-pong (8 B one-way time; 32 MB bandwidth).
Writes results/tables/E3_<platform>_model.csv and results/figures/e3_model_<platform>.png.
"""
import argparse
import csv
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
RANKS_PER_NODE = 64  # multi-node runs place 64 ranks per node (ppr)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--platform", default="vanda")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    tables, out = pathlib.Path(args.tables), pathlib.Path(args.out)

    e0 = [r for r in read(tables / "E0.csv") if r.get("platform") == args.platform]
    link = {}
    for name in ("intra", "inter"):
        rows = {int(r["bytes"]): float(r["t_oneway"]) for r in e0
                if r.get("kind") == "pingpong" and r["log"] == f"pingpong_{name}.log"}
        big = max(rows)
        link[name] = (rows[min(rows)], big / rows[big])  # alpha (s), beta (bytes/s)
    conc = [float(r["total_GBps"]) * 1e9 for r in e0
            if r.get("kind") == "pingpong_concurrent" and r["log"].startswith("conc_inter") and int(r["bytes"]) == 33554432]
    b_node = max(conc)

    rows = [r for r in read(tables / "E3.csv") if r.get("platform") == args.platform]
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from plot import median_runs  # median of repeated runs per rank count
    pick = lambda key: median_runs(rows, "e3_p", key)
    mpi_runs, ring_runs, dense_runs = pick("mpi"), pick("ring"), pick("dense")
    ring_by_p = {int(r["ranks"]): r for r in ring_runs}
    one = [r for r in mpi_runs + ring_runs if int(r["ranks"]) == 1]
    c1 = sum(float(r["t_compute"]) for r in one) / len(one)
    o1 = sum(float(r["t_optim"]) for r in one) / len(one)
    r0 = one[0]
    B, H, F = int(r0["batch"]), int(r0["hidden"]), int(r0["ffn"])
    E, k, L = int(r0["experts"]), int(r0["topk"]), int(r0["blocks"])
    s = 4 if r0["real"] == "float" else 8
    cf = float(r0["capacity"])
    dense_f = 784 * H + L * H * E + H * 10        # forward flops per token / 2
    expert_f = L * k * 2 * H * F
    c_dense, c_expert = c1 * dense_f / (dense_f + expert_f), c1 * expert_f / (dense_f + expert_f)
    n_dense = 784 * H + H + L * (H * E + E) + H * 10 + 10
    n_expert_rank = lambda ep: L * (E // ep) * (2 * H * F + F + H)
    tau = o1 / (n_dense + n_expert_rank(1))       # Adam seconds per parameter

    # f(ranks on one node): dense compute efficiency, measured on one node
    d1 = float(next(r["t_compute"] for r in dense_runs if int(r["ranks"]) == 1))
    node_eff = {int(r["ranks"]): d1 / (int(r["ranks"]) * float(r["t_compute"]))
                for r in dense_runs if int(r["ranks"]) <= RANKS_PER_NODE}

    def ring(q, n_values, alpha, beta):
        return 0.0 if q <= 1 else 2 * (q - 1) * alpha + 2 * (q - 1) / q * n_values * s / beta

    out_rows = []
    (ai, bi), (ax, bx) = link["intra"], link["inter"]
    for r in mpi_runs:
        P, ep = int(r["ranks"]), int(r["ep"])
        D = P // ep
        nodes = max(1, math.ceil(P / RANKS_PER_NODE))
        f = node_eff.get(min(P, RANKS_PER_NODE), 1.0)  # reported only
        imb = float(r["imbalance"]) if float(r["imbalance"]) > 0 else 1.0
        rank_imb = 1.0 + (min(imb, cf) - 1.0) * (ep - 1) / max(E - 1, 1)
        t_comp = c_dense / P + c_expert * rank_imb / P
        t_a2a = 0.0 if ep == 1 else 4 * L * ((ep - 1) * ai + (B / P) * k * (H + 2) * s * (ep - 1) / ep / bi)
        a, b = (ax, bx) if nodes > 1 else (ai, bi)
        ar_dense = ring(P, n_dense, a, b)
        groups_crossing = ep if (nodes > 1 and D > 1) else 1   # one expert-DP group per expert slot
        ar_exp_iso = ring(D, n_expert_rank(ep), a, b)
        ar_exp_shared = ring(D, n_expert_rank(ep), a, min(b, b_node / groups_crossing) if nodes > 1 else b)
        t_opt = (n_dense + n_expert_rank(ep)) * tau
        rr = ring_by_p.get(P, r)
        out_rows.append({
            "ranks": P, "ep": ep, "D": D, "nodes": nodes, "imbalance": round(imb, 3), "rank_imbalance": round(rank_imb, 3),
            "dense_compute_eff_same_ranks_per_node": round(f, 3),
            "pred_compute_s": t_comp, "meas_compute_plus_wait_s": float(r["t_compute"]) + float(r["t_a2a_wait"]),
            "pred_alltoall_s": t_a2a, "meas_alltoall_s": float(r["t_a2a"]),
            "pred_allreduce_isolated_s": ar_dense + ar_exp_iso, "pred_allreduce_shared_s": ar_dense + ar_exp_shared,
            "meas_allreduce_ring_s": float(rr["t_comm"]), "meas_allreduce_mpi_s": float(r["t_comm"]),
            "pred_optim_s": t_opt, "meas_optim_s": float(r["t_optim"]),
            "pred_step_s": t_comp + t_a2a + ar_dense + ar_exp_shared + t_opt,
            "meas_step_ring_s": float(rr["t_step_mean"]), "meas_step_mpi_s": float(r["t_step_mean"]),
        })
    with open(tables / f"E3_{args.platform}_model.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    x = [r["ranks"] for r in out_rows]
    panels = [
        ("time per step", [("meas_step_ring_s", "measured, ring all-reduce", "o-"),
                           ("meas_step_mpi_s", "measured, MPI_Allreduce", "s-"),
                           ("pred_step_s", "model (ring, shared link)", "--")]),
        ("compute incl. imbalance wait", [("meas_compute_plus_wait_s", "measured", "o-"),
                                          ("pred_compute_s", "model", "--")]),
        ("gradient all-reduce", [("meas_allreduce_ring_s", "measured, ring", "o-"),
                                 ("meas_allreduce_mpi_s", "measured, MPI_Allreduce", "s-"),
                                 ("pred_allreduce_isolated_s", "model, isolated pair", ":"),
                                 ("pred_allreduce_shared_s", "model, shared node link", "--")]),
    ]
    for ax_, (title, lines) in zip(axes, panels):
        for i, (col, label, ls) in enumerate(lines):
            xs = [xx for xx, r in zip(x, out_rows) if r[col] > 0]
            ys = [r[col] for r in out_rows if r[col] > 0]
            ax_.plot(xs, ys, ls, color=SERIES[i], lw=1.5, ms=4, label=label)
        ax_.set_xscale("log", base=2)
        ax_.set_yscale("log")
        ax_.set_xlabel("MPI ranks (E3, MoE)")
        ax_.set_ylabel("seconds")
        ax_.set_title(title, fontsize=9)
        ax_.legend(fontsize=7)
        ax_.grid(True, color="#e4e4e0", linewidth=0.6)
        for side in ("top", "right"):
            ax_.spines[side].set_visible(False)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"e3_model_{args.platform}.png", dpi=200)
    plt.close(fig)
    print(f"alpha/beta: { {k_: (f'{v[0] * 1e6:.1f} us', f'{v[1] / 1e9:.2f} GB/s') for k_, v in link.items()} }, "
          f"node link {b_node / 1e9:.1f} GB/s, Adam {tau * 1e9:.2f} ns/param, node eff {node_eff}")
    for r in out_rows:
        print(f"P={r['ranks']:4d} step pred {r['pred_step_s']:.3f} meas ring {r['meas_step_ring_s']:.3f} "
              f"mpi {r['meas_step_mpi_s']:.3f} | comp {r['pred_compute_s']:.3f}/{r['meas_compute_plus_wait_s']:.3f} | "
              f"a2a {r['pred_alltoall_s'] * 1e3:.1f}/{r['meas_alltoall_s'] * 1e3:.1f} ms | ar iso {r['pred_allreduce_isolated_s'] * 1e3:.1f} "
              f"shared {r['pred_allreduce_shared_s'] * 1e3:.1f} ring {r['meas_allreduce_ring_s'] * 1e3:.1f} mpi {r['meas_allreduce_mpi_s'] * 1e3:.1f} ms | "
              f"opt {r['pred_optim_s'] * 1e3:.1f}/{r['meas_optim_s'] * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
