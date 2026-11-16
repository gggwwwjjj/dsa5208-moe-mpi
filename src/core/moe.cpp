#include "moe.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../backend/backend.hpp"

namespace {

// identity exchange for the serial build / a single expert-parallel rank
void exchange_rows(const ExpertComm& comm, const rvec& send, const std::vector<int>& send_counts,
                   int row_len, rvec& recv, std::vector<int>& recv_counts, bool counts_known) {
    if (comm.exchange) {
        comm.exchange(send, send_counts, row_len, recv, recv_counts, counts_known);
    } else {
        recv = send;
        recv_counts = send_counts;
    }
}

}  // namespace

MoELayer::MoELayer(ParamStore& dense, ParamStore& experts, const MoEConfig& cfg, const ExpertComm& comm,
                   std::uint64_t seed, std::uint64_t layer_id)
    : cfg_(cfg), comm_(&comm), experts_per_rank_(cfg.experts / std::max(comm.ep_size, 1)),
      router_(dense, cfg.hidden, cfg.experts) {
    if (cfg.experts % comm.ep_size != 0) throw std::runtime_error("experts must be divisible by the expert-parallel size");
    if (cfg.topk < 1 || cfg.topk > cfg.experts) throw std::runtime_error("need 1 <= topk <= experts");
    for (int i = 0; i < experts_per_rank_; ++i) {
        up_.emplace_back(experts, cfg.hidden, cfg.ffn);
        down_.emplace_back(experts, cfg.ffn, cfg.hidden);
    }
    // small router weights: near-uniform routing at the start
    router_.init(seed, 1000 + 100 * layer_id, 0.1);
    for (int i = 0; i < experts_per_rank_; ++i) {
        // seeds depend on the GLOBAL expert id, so an expert has the same
        // initial weights whichever rank owns it (serial == expert-parallel)
        const std::uint64_t e = static_cast<std::uint64_t>(comm.ep_rank * experts_per_rank_ + i);
        up_[static_cast<std::size_t>(i)].init(seed, 2000000 + 10000 * layer_id + 2 * e);
        down_[static_cast<std::size_t>(i)].init(seed, 2000000 + 10000 * layer_id + 2 * e + 1);
    }
    xe_.resize(static_cast<std::size_t>(experts_per_rank_));
    ue_.resize(static_cast<std::size_t>(experts_per_rank_));
    re_.resize(static_cast<std::size_t>(experts_per_rank_));
}

void MoELayer::forward(const real* h, int rows, real* out, long pos_offset, long batch, bool train) {
    const int H = cfg_.hidden, E = cfg_.experts, K = cfg_.topk, F = cfg_.ffn;
    const std::size_t Hs = static_cast<std::size_t>(H);
    rows_ = rows;
    batch_ = batch;
    train_ = train;

    // 1. router probabilities. The input is mean-centred first: h comes out of
    // ReLU, so every row shares a large positive component, and without
    // centring that component pushes all tokens to the same experts (the
    // role LayerNorm plays before the router in Switch / GShard).
    hc_.resize(static_cast<std::size_t>(rows) * Hs);
    for (int t = 0; t < rows; ++t) {
        const real* src = h + static_cast<std::size_t>(t) * Hs;
        real* dst = &hc_[static_cast<std::size_t>(t) * Hs];
        double mean = 0;
        for (int c = 0; c < H; ++c) mean += src[c];
        mean /= H;
        for (int c = 0; c < H; ++c) dst[c] = static_cast<real>(src[c] - mean);
    }
    prob_.resize(static_cast<std::size_t>(rows) * E);
    router_.forward(hc_.data(), rows, prob_.data());
    for (int t = 0; t < rows; ++t) {
        real* p = &prob_[static_cast<std::size_t>(t) * E];
        real m = *std::max_element(p, p + E);
        double s = 0;
        for (int e = 0; e < E; ++e) s += std::exp(static_cast<double>(p[e] - m));
        for (int e = 0; e < E; ++e) p[e] = static_cast<real>(std::exp(static_cast<double>(p[e] - m)) / s);
    }

    // 2. top-k (ties -> lower expert id) and gates
    choice_.resize(static_cast<std::size_t>(rows) * K);
    gate_.resize(static_cast<std::size_t>(rows) * K);
    gate_sum_.resize(static_cast<std::size_t>(rows));
    std::vector<char> taken(static_cast<std::size_t>(E));
    for (int t = 0; t < rows; ++t) {
        const real* p = &prob_[static_cast<std::size_t>(t) * E];
        std::fill(taken.begin(), taken.end(), 0);
        double s = 0;
        for (int j = 0; j < K; ++j) {
            int best = -1;
            for (int e = 0; e < E; ++e)
                if (!taken[static_cast<std::size_t>(e)] && (best < 0 || p[e] > p[best])) best = e;
            taken[static_cast<std::size_t>(best)] = 1;
            choice_[static_cast<std::size_t>(t) * K + j] = best;
            s += p[best];
        }
        gate_sum_[static_cast<std::size_t>(t)] = static_cast<real>(K == 1 ? 1.0 : s);
        for (int j = 0; j < K; ++j) {
            const real pj = p[choice_[static_cast<std::size_t>(t) * K + j]];
            gate_[static_cast<std::size_t>(t) * K + j] = K == 1 ? pj : static_cast<real>(pj / s);
        }
    }

    // 3. load statistics and aux loss from GLOBAL counts (same value on every rank)
    aux_loss_ = 0;
    if (train) {
        std::vector<double> cnt(2 * static_cast<std::size_t>(E), 0.0);
        for (std::size_t a = 0; a < choice_.size(); ++a) cnt[static_cast<std::size_t>(choice_[a])] += 1;
        for (int t = 0; t < rows; ++t)
            for (int e = 0; e < E; ++e)
                cnt[static_cast<std::size_t>(E + e)] += prob_[static_cast<std::size_t>(t) * E + e];
        if (comm_->sum_counts) comm_->sum_counts(cnt.data(), 2 * E);
        f_.assign(static_cast<std::size_t>(E), 0.0);
        double max_load = 0;
        for (int e = 0; e < E; ++e) {
            f_[static_cast<std::size_t>(e)] = cnt[static_cast<std::size_t>(e)] / (static_cast<double>(batch) * K);
            aux_loss_ += f_[static_cast<std::size_t>(e)] * cnt[static_cast<std::size_t>(E + e)] / static_cast<double>(batch);
            max_load = std::max(max_load, cnt[static_cast<std::size_t>(e)]);
        }
        aux_loss_ *= cfg_.aux_weight * E;
        const double mean_load = static_cast<double>(batch) * K / E;
        double var = 0;
        for (int e = 0; e < E; ++e) {
            const double d = cnt[static_cast<std::size_t>(e)] - mean_load;
            var += d * d;
        }
        stats_.max_load += max_load;
        stats_.mean_load += mean_load;
        stats_.cv_load += std::sqrt(var / E) / mean_load;
        stats_.aux_loss += aux_loss_;
        ++stats_.steps;
    }

    // 4. dispatch: assignments ordered by the rank that owns their expert
    const int R = comm_->ep_size;
    const int n_assign = rows * K;
    send_counts_.assign(static_cast<std::size_t>(R), 0);
    for (int a = 0; a < n_assign; ++a) ++send_counts_[static_cast<std::size_t>(owner(choice_[static_cast<std::size_t>(a)]))];
    std::vector<int> offset(static_cast<std::size_t>(R) + 1, 0);
    for (int r = 0; r < R; ++r) offset[static_cast<std::size_t>(r) + 1] = offset[static_cast<std::size_t>(r)] + send_counts_[static_cast<std::size_t>(r)];
    send_order_.assign(static_cast<std::size_t>(n_assign), 0);
    {
        std::vector<int> next(offset.begin(), offset.end() - 1);
        for (int a = 0; a < n_assign; ++a)
            send_order_[static_cast<std::size_t>(next[static_cast<std::size_t>(owner(choice_[static_cast<std::size_t>(a)]))]++)] = a;
    }
    const int in_len = H + 2;  // [global position, expert id, x_0 .. x_{H-1}]
    rvec send(static_cast<std::size_t>(n_assign) * static_cast<std::size_t>(in_len));
    for (int i = 0; i < n_assign; ++i) {
        const int a = send_order_[static_cast<std::size_t>(i)], t = a / K;
        real* row = &send[static_cast<std::size_t>(i) * static_cast<std::size_t>(in_len)];
        row[0] = static_cast<real>(pos_offset + t);
        row[1] = static_cast<real>(choice_[static_cast<std::size_t>(a)]);
        std::copy(h + static_cast<std::size_t>(t) * Hs, h + static_cast<std::size_t>(t + 1) * Hs, row + 2);
    }
    exchange_rows(*comm_, send, send_counts_, in_len, recv_, recv_counts_, false);  // dispatch
    const int n_recv = static_cast<int>(recv_.size() / static_cast<std::size_t>(in_len));

    // 5. expert owner: capacity by global position within each token group (plan D16)
    for (auto& v : expert_rows_) v.clear();
    expert_rows_.assign(static_cast<std::size_t>(experts_per_rank_), {});
    for (int i = 0; i < n_recv; ++i) {
        const int e = static_cast<int>(recv_[static_cast<std::size_t>(i) * static_cast<std::size_t>(in_len) + 1]);
        expert_rows_[static_cast<std::size_t>(e % experts_per_rank_)].push_back(i);
    }
    const bool capped = train && cfg_.capacity_factor > 0;
    const long group_tokens = std::max(1L, batch / std::max(cfg_.groups, 1));
    const long capacity = static_cast<long>(std::ceil(cfg_.capacity_factor * K * static_cast<double>(group_tokens) / E));
    long kept_total = 0;
    for (auto& rows_e : expert_rows_) {
        auto pos_of = [&](int i) { return static_cast<long>(recv_[static_cast<std::size_t>(i) * static_cast<std::size_t>(in_len)]); };
        std::stable_sort(rows_e.begin(), rows_e.end(), [&](int a, int b) { return pos_of(a) < pos_of(b); });
        if (capped) {
            std::vector<int> kept;
            long group = -1, used = 0;
            for (int i : rows_e) {
                const long g = pos_of(i) / group_tokens;
                if (g != group) { group = g; used = 0; }
                if (used < capacity) { kept.push_back(i); ++used; }
            }
            rows_e.swap(kept);
        }
        kept_total += static_cast<long>(rows_e.size());
    }
    if (train) {
        stats_.assignments += n_recv;
        stats_.dropped += static_cast<double>(n_recv - kept_total);
    }

    // expert FFNs on their kept rows; results go back as [kept, y_0 .. y_{H-1}]
    const int out_len = H + 1;
    rvec back_send(static_cast<std::size_t>(n_recv) * static_cast<std::size_t>(out_len), real(0));
    for (int le = 0; le < experts_per_rank_; ++le) {
        const auto& rows_e = expert_rows_[static_cast<std::size_t>(le)];
        const int n = static_cast<int>(rows_e.size());
        auto& X = xe_[static_cast<std::size_t>(le)];
        auto& U = ue_[static_cast<std::size_t>(le)];
        auto& Rl = re_[static_cast<std::size_t>(le)];
        X.resize(static_cast<std::size_t>(n) * Hs);
        U.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(F));
        Rl.resize(U.size());
        for (int r = 0; r < n; ++r) {
            const real* src = &recv_[static_cast<std::size_t>(rows_e[static_cast<std::size_t>(r)]) * static_cast<std::size_t>(in_len) + 2];
            std::copy(src, src + Hs, &X[static_cast<std::size_t>(r) * Hs]);
        }
        if (n == 0) continue;
        rvec Y(static_cast<std::size_t>(n) * Hs);
        up_[static_cast<std::size_t>(le)].forward(X.data(), n, U.data());
        backend::relu_forward(U.data(), Rl.data(), U.size());
        down_[static_cast<std::size_t>(le)].forward(Rl.data(), n, Y.data());
        for (int r = 0; r < n; ++r) {
            real* dst = &back_send[static_cast<std::size_t>(rows_e[static_cast<std::size_t>(r)]) * static_cast<std::size_t>(out_len)];
            dst[0] = real(1);
            std::copy(&Y[static_cast<std::size_t>(r) * Hs], &Y[static_cast<std::size_t>(r + 1) * Hs], dst + 1);
        }
    }
    // combine: the reverse exchange, so this rank receives what it sent in dispatch
    std::vector<int> back_counts = send_counts_;
    exchange_rows(*comm_, back_send, recv_counts_, out_len, back_, back_counts, true);

    // 6. combine at the token owner: out = h + sum of gated expert outputs
    std::copy(h, h + static_cast<std::size_t>(rows) * Hs, out);
    for (int i = 0; i < n_assign; ++i) {
        const real* row = &back_[static_cast<std::size_t>(i) * static_cast<std::size_t>(out_len)];
        if (row[0] == real(0)) continue;  // dropped: residual only
        const int a = send_order_[static_cast<std::size_t>(i)], t = a / K;
        const real g = gate_[static_cast<std::size_t>(a)];
        real* o = out + static_cast<std::size_t>(t) * Hs;
        for (int c = 0; c < H; ++c) o[c] += g * row[1 + c];
    }
}

void MoELayer::backward(const real* /*h: the centred copy hc_ is used*/, const real* dout, real* dh) {
    const int H = cfg_.hidden, E = cfg_.experts, K = cfg_.topk, F = cfg_.ffn;
    const std::size_t Hs = static_cast<std::size_t>(H);
    const int rows = rows_, n_assign = rows * K, out_len = H + 1;
    std::copy(dout, dout + static_cast<std::size_t>(rows) * Hs, dh);  // residual path

    // token owner: gradient for each expert output (g * dout) and for each gate (<dout, y>)
    rvec dy(static_cast<std::size_t>(n_assign) * Hs, real(0));
    std::vector<double> dg(static_cast<std::size_t>(n_assign), 0.0);
    for (int i = 0; i < n_assign; ++i) {
        const real* row = &back_[static_cast<std::size_t>(i) * static_cast<std::size_t>(out_len)];
        if (row[0] == real(0)) continue;
        const int a = send_order_[static_cast<std::size_t>(i)], t = a / K;
        const real g = gate_[static_cast<std::size_t>(a)];
        const real* d = dout + static_cast<std::size_t>(t) * Hs;
        double dot = 0;
        for (int c = 0; c < H; ++c) {
            dy[static_cast<std::size_t>(i) * Hs + static_cast<std::size_t>(c)] = g * d[c];
            dot += static_cast<double>(d[c]) * row[1 + c];
        }
        dg[static_cast<std::size_t>(a)] = dot;
    }
    // dy travels like dispatch, dx like combine: all counts are known
    rvec dy_recv;
    std::vector<int> counts = recv_counts_;
    exchange_rows(*comm_, dy, send_counts_, H, dy_recv, counts, true);

    // expert owner: backward through each expert FFN
    const int n_recv = static_cast<int>(dy_recv.size() / Hs);
    rvec dx_send(static_cast<std::size_t>(n_recv) * Hs, real(0));
    for (int le = 0; le < experts_per_rank_; ++le) {
        const auto& rows_e = expert_rows_[static_cast<std::size_t>(le)];
        const int n = static_cast<int>(rows_e.size());
        if (n == 0) continue;
        rvec dY(static_cast<std::size_t>(n) * Hs), dR(static_cast<std::size_t>(n) * static_cast<std::size_t>(F)),
            dX(static_cast<std::size_t>(n) * Hs);
        for (int r = 0; r < n; ++r) {
            const real* src = &dy_recv[static_cast<std::size_t>(rows_e[static_cast<std::size_t>(r)]) * Hs];
            std::copy(src, src + Hs, &dY[static_cast<std::size_t>(r) * Hs]);
        }
        down_[static_cast<std::size_t>(le)].backward(re_[static_cast<std::size_t>(le)].data(), dY.data(), n, dR.data());
        backend::relu_backward(ue_[static_cast<std::size_t>(le)].data(), dR.data(), dR.data(), dR.size());
        up_[static_cast<std::size_t>(le)].backward(xe_[static_cast<std::size_t>(le)].data(), dR.data(), n, dX.data());
        for (int r = 0; r < n; ++r)
            std::copy(&dX[static_cast<std::size_t>(r) * Hs], &dX[static_cast<std::size_t>(r + 1) * Hs],
                      &dx_send[static_cast<std::size_t>(rows_e[static_cast<std::size_t>(r)]) * Hs]);
    }
    rvec dx_back;
    counts = send_counts_;
    exchange_rows(*comm_, dx_send, recv_counts_, H, dx_back, counts, true);
    for (int i = 0; i < n_assign; ++i) {
        const int t = send_order_[static_cast<std::size_t>(i)] / K;
        real* d = dh + static_cast<std::size_t>(t) * Hs;
        const real* s = &dx_back[static_cast<std::size_t>(i) * Hs];
        for (int c = 0; c < H; ++c) d[c] += s[c];
    }

    // router: gates -> probabilities -> logits
    rvec dlogits(static_cast<std::size_t>(rows) * E, real(0));
    std::vector<double> dp(static_cast<std::size_t>(E));
    const double aux_scale = train_ ? cfg_.aux_weight * E / static_cast<double>(batch_) : 0.0;
    for (int t = 0; t < rows; ++t) {
        const real* p = &prob_[static_cast<std::size_t>(t) * E];
        std::fill(dp.begin(), dp.end(), 0.0);
        const int* ch = &choice_[static_cast<std::size_t>(t) * K];
        const double* g = &dg[static_cast<std::size_t>(t) * K];
        if (K == 1) {
            dp[static_cast<std::size_t>(ch[0])] = g[0];  // g = p
        } else {
            // g_j = p_j / S, S = sum of chosen p: dp_i = dg_i / S - (sum_j dg_j p_j) / S^2
            const double S = gate_sum_[static_cast<std::size_t>(t)];
            double s = 0;
            for (int j = 0; j < K; ++j) s += g[j] * p[ch[j]];
            for (int j = 0; j < K; ++j) dp[static_cast<std::size_t>(ch[j])] += g[j] / S - s / (S * S);
        }
        if (aux_scale > 0)
            for (int e = 0; e < E; ++e) dp[static_cast<std::size_t>(e)] += aux_scale * f_[static_cast<std::size_t>(e)];
        double dot = 0;
        for (int e = 0; e < E; ++e) dot += p[e] * dp[static_cast<std::size_t>(e)];
        for (int e = 0; e < E; ++e)
            dlogits[static_cast<std::size_t>(t) * E + e] = static_cast<real>(p[e] * (dp[static_cast<std::size_t>(e)] - dot));
    }
    // router saw hc = h - mean(h): the gradient w.r.t. h is the centred gradient w.r.t. hc
    rvec dtmp(static_cast<std::size_t>(rows) * Hs);
    router_.backward(hc_.data(), dlogits.data(), rows, dtmp.data());
    for (int t = 0; t < rows; ++t) {
        const real* g = &dtmp[static_cast<std::size_t>(t) * Hs];
        real* d = dh + static_cast<std::size_t>(t) * Hs;
        double mean = 0;
        for (int c = 0; c < H; ++c) mean += g[c];
        mean /= H;
        for (int c = 0; c < H; ++c) d[c] += static_cast<real>(g[c] - mean);
    }
}
