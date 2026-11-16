#pragma once
// Mixture-of-experts block (plan D1, D2, D12, D16):
//   out = h + sum_{e in TopK(h)} g_e(h) * FFN_e(h)
// router = Linear H->E on the mean-centred input + softmax; top-k by
// probability, ties -> lower expert id;
// k >= 2: gates renormalised over the chosen experts; k = 1: g = p (Switch),
// so the router still receives a gradient. Auxiliary load-balancing loss
// aux = alpha * E * sum_e f_e P_e (Switch Transformer). Capacity
// C = ceil(cf * k * T_group / E) assignments per expert per token group; over
// capacity, the assignments with the smallest global batch position are kept.
//
// Tokens travel to the rank that owns their expert through an exchange
// (identity in the serial build, MPI_Alltoallv with expert parallelism), so
// this file contains no MPI and the same code runs in both builds.
#include <cstdint>
#include <functional>
#include <vector>

#include "layers.hpp"

struct MoEConfig {
    int hidden = 512;
    int ffn = 1024;
    int experts = 16;
    int topk = 2;
    double capacity_factor = 1.25;  // <= 0: no capacity limit
    double aux_weight = 0.1;
    int groups = 1;                 // token groups for capacity (D in plan D16)
};

// Communication used by the MoE blocks. Default-constructed = serial.
struct ExpertComm {
    int ep_size = 1;  // ranks the experts are spread over (expert-parallel group)
    int ep_rank = 0;
    // All-to-all of fixed-length rows inside the expert-parallel group.
    // send holds rows ordered by destination rank, send_counts[r] rows for r.
    // counts_known: recv_counts already holds the counts (the exchanges after
    // dispatch), so no counts all-to-all is needed.
    std::function<void(const rvec& send, const std::vector<int>& send_counts, int row_len,
                       rvec& recv, std::vector<int>& recv_counts, bool counts_known)>
        exchange;
    // Sum over all data-parallel ranks (routing counts for the aux loss).
    std::function<void(double*, int)> sum_counts;
};

struct MoEStats {
    double assignments = 0;  // token-expert assignments received by local experts
    double dropped = 0;      // of those, dropped by the capacity limit
    double max_load = 0;     // largest per-expert load (global counts), summed over steps
    double mean_load = 0;    // mean per-expert load, summed over steps
    double cv_load = 0;      // coefficient of variation (std / mean) of per-expert load, summed over steps
    double aux_loss = 0;     // summed over steps
    long steps = 0;
};

class MoELayer {
public:
    MoELayer(ParamStore& dense, ParamStore& experts, const MoEConfig& cfg, const ExpertComm& comm,
             std::uint64_t seed, std::uint64_t layer_id);

    // h: rows x H input; out: rows x H output. pos_offset: global batch
    // position of row 0; batch: global batch size. train = false disables
    // capacity and the aux loss (evaluation).
    void forward(const real* h, int rows, real* out, long pos_offset, long batch, bool train);
    // dout: gradient w.r.t. out (already scaled by 1/B); dh: gradient w.r.t. h.
    void backward(const real* h, const real* dout, real* dh);

    double aux_loss() const { return aux_loss_; }
    const MoEStats& stats() const { return stats_; }
    // Expert ids chosen for each row in the last forward, rows x k (E11 routing heatmap).
    const std::vector<int>& last_choice() const { return choice_; }

private:
    int owner(int e) const { return e / experts_per_rank_; }

    MoEConfig cfg_;
    const ExpertComm* comm_;
    int experts_per_rank_;
    Linear router_;
    std::vector<Linear> up_, down_;  // local experts only

    // forward state
    int rows_ = 0;
    long batch_ = 0;
    bool train_ = false;
    rvec hc_;            // router input: h minus its per-row mean, rows x H
    rvec prob_;          // rows x E
    std::vector<int> choice_;         // rows x k expert ids
    rvec gate_;          // rows x k
    rvec gate_sum_;      // rows (sum of chosen probabilities, k >= 2)
    std::vector<int> send_order_;     // assignment index (t*k + j) in send order
    std::vector<int> send_counts_, recv_counts_;
    rvec recv_;          // received rows: [pos, expert, x_0..x_{H-1}]
    std::vector<std::vector<int>> expert_rows_;  // per local expert: kept received-row indices
    std::vector<rvec> xe_, ue_, re_;  // per local expert activations
    rvec back_;          // returned rows in send order: [kept, y_0..y_{H-1}]
    std::vector<double> f_;           // global fraction of assignments per expert
    double aux_loss_ = 0;
    MoEStats stats_;
};
