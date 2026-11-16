#include "model.hpp"

#include <cmath>

#include "../backend/backend.hpp"
#include "rng.hpp"

Model::Model(const ModelConfig& cfg, std::uint64_t seed, const ExpertComm& comm)
    : cfg_(cfg), comm_(comm), in_(store_, cfg.input, cfg.hidden), out_(store_, cfg.hidden, cfg.classes) {
    in_.init(seed, 1);
    out_.init(seed, 2);
    if (cfg.experts <= 1) {
        for (int l = 0; l < cfg.blocks; ++l) {
            up_.emplace_back(store_, cfg.hidden, cfg.ffn);
            down_.emplace_back(store_, cfg.ffn, cfg.hidden);
        }
        for (int l = 0; l < cfg.blocks; ++l) {
            up_[static_cast<std::size_t>(l)].init(seed, 10 + 2 * static_cast<std::uint64_t>(l));
            down_[static_cast<std::size_t>(l)].init(seed, 11 + 2 * static_cast<std::uint64_t>(l));
        }
    } else {
        MoEConfig m;
        m.hidden = cfg.hidden;
        m.ffn = cfg.ffn;
        m.experts = cfg.experts;
        m.topk = cfg.topk;
        m.capacity_factor = cfg.capacity_factor;
        m.aux_weight = cfg.aux_weight;
        m.groups = cfg.groups;
        for (int l = 0; l < cfg.blocks; ++l)
            moe_.push_back(std::make_unique<MoELayer>(store_, expert_store_, m, comm_, seed, static_cast<std::uint64_t>(l)));
    }
    h_.resize(static_cast<std::size_t>(cfg.blocks) + 1);
    u_.resize(up_.size());
    r_.resize(up_.size());
}

double Model::flops_per_sample() const {
    // per block: dense FFN, or router + k expert FFNs
    const double ffn = 2.0 * cfg_.hidden * cfg_.ffn;
    const double block = is_moe() ? static_cast<double>(cfg_.hidden) * cfg_.experts + cfg_.topk * ffn : ffn;
    const double fwd = 2.0 * (static_cast<double>(cfg_.input) * cfg_.hidden + cfg_.blocks * block +
                              static_cast<double>(cfg_.hidden) * cfg_.classes);
    return 3.0 * fwd;  // backward = two GEMMs per forward GEMM
}

void Model::forward(const real* X, int rows, long pos_offset, long batch, bool train) {
    const std::size_t H = static_cast<std::size_t>(cfg_.hidden), F = static_cast<std::size_t>(cfg_.ffn);
    const std::size_t R = static_cast<std::size_t>(rows);
    a_in_.resize(R * H);
    for (auto& v : h_) v.resize(R * H);
    for (auto& v : u_) v.resize(R * F);
    for (auto& v : r_) v.resize(R * F);
    logits_.resize(R * static_cast<std::size_t>(cfg_.classes));

    in_.forward(X, rows, a_in_.data());
    backend::relu_forward(a_in_.data(), h_[0].data(), R * H);
    for (std::size_t l = 0; l < static_cast<std::size_t>(cfg_.blocks); ++l) {
        if (is_moe()) {
            moe_[l]->forward(h_[l].data(), rows, h_[l + 1].data(), pos_offset, batch, train);
        } else {
            up_[l].forward(h_[l].data(), rows, u_[l].data());
            backend::relu_forward(u_[l].data(), r_[l].data(), R * F);
            down_[l].forward(r_[l].data(), rows, h_[l + 1].data());
            backend::add_inplace(h_[l + 1].data(), h_[l].data(), R * H);  // residual
        }
    }
    out_.forward(h_.back().data(), rows, logits_.data());
}

namespace {

// Softmax cross-entropy over one row; writes (p - onehot) * scale into drow
// when drow != nullptr. Returns the row's loss.
double softmax_ce_row(const real* z, int classes, int label, real scale, real* drow, bool* hit) {
    real zmax = z[0];
    int arg = 0;
    for (int c = 1; c < classes; ++c)
        if (z[c] > zmax) { zmax = z[c]; arg = c; }
    double denom = 0;
    for (int c = 0; c < classes; ++c) denom += std::exp(static_cast<double>(z[c] - zmax));
    if (drow) {
        for (int c = 0; c < classes; ++c) {
            double p = std::exp(static_cast<double>(z[c] - zmax)) / denom;
            drow[c] = static_cast<real>((p - (c == label ? 1.0 : 0.0)) * scale);
        }
    }
    *hit = (arg == label);
    return std::log(denom) - static_cast<double>(z[label] - zmax);
}

}  // namespace

StepStats Model::loss(const std::uint8_t* y, int rows, real grad_scale, bool want_grad) {
    const int C = cfg_.classes;
    if (want_grad) dlogits_.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(C));
    StepStats st;
    st.rows = rows;
    double loss = 0;
    long correct = 0;
    // loss is only logged, so the thread-dependent summation order does not matter;
    // the gradients are written per row and are thread-independent
#pragma omp parallel for reduction(+ : loss, correct) schedule(static) if (rows > 256)
    for (int i = 0; i < rows; ++i) {
        bool hit = false;
        loss += softmax_ce_row(&logits_[static_cast<std::size_t>(i) * C], C, y[i], grad_scale,
                               want_grad ? &dlogits_[static_cast<std::size_t>(i) * C] : nullptr, &hit);
        correct += hit ? 1 : 0;
    }
    st.loss_sum = loss;
    st.correct = correct;
    for (const auto& m : moe_) st.aux += m->aux_loss();
    return st;
}

StepStats Model::forward_backward(const real* X, const std::uint8_t* y, int rows, real grad_scale, long pos_offset,
                                  long batch) {
    forward(X, rows, pos_offset, batch, true);
    StepStats st = loss(y, rows, grad_scale, true);
    const std::size_t H = static_cast<std::size_t>(cfg_.hidden), F = static_cast<std::size_t>(cfg_.ffn);
    const std::size_t R = static_cast<std::size_t>(rows);
    dh_.resize(R * H);
    dr_.resize(R * F);
    dtmp_.resize(R * H);
    out_.backward(h_.back().data(), dlogits_.data(), rows, dh_.data());
    for (std::size_t l = static_cast<std::size_t>(cfg_.blocks); l-- > 0;) {
        if (is_moe()) {
            // the MoE block writes the full gradient w.r.t. its input (residual included)
            moe_[l]->backward(h_[l].data(), dh_.data(), dtmp_.data());
            dh_.swap(dtmp_);
        } else {
            // h_{l+1} = h_l + down(relu(up(h_l))): dh reaches h_l directly and through the FFN
            down_[l].backward(r_[l].data(), dh_.data(), rows, dr_.data());
            backend::relu_backward(u_[l].data(), dr_.data(), dr_.data(), R * F);
            up_[l].backward(h_[l].data(), dr_.data(), rows, dtmp_.data());
            backend::add_inplace(dh_.data(), dtmp_.data(), R * H);
        }
    }
    backend::relu_backward(a_in_.data(), dh_.data(), dh_.data(), R * H);
    in_.backward(X, dh_.data(), rows, nullptr);  // no gradient w.r.t. the input data
    return st;
}

StepStats Model::training_loss(const real* X, const std::uint8_t* y, int rows, long pos_offset, long batch) {
    forward(X, rows, pos_offset, batch, true);
    return loss(y, rows, real(0), false);
}

StepStats Model::evaluate(const real* X, const std::uint8_t* y, int rows) {
    forward(X, rows, 0, rows, false);
    return loss(y, rows, real(0), false);
}
