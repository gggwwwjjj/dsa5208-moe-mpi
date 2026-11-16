#include "train.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#include "optim.hpp"
#include "rng.hpp"

void DatasetSource::fill(long step, long batch, long first, long rows, real* X, std::uint8_t* y) {
    const long per_epoch = steps_per_epoch(batch);
    const long epoch = step / per_epoch;
    if (epoch != perm_epoch_) {
        perm_ = epoch_permutation(data_.n, seed_, epoch);
        perm_epoch_ = epoch;
    }
    const long base = (step % per_epoch) * batch + first;
    for (long i = 0; i < rows; ++i) {
        const long s = perm_[static_cast<std::size_t>(base + i)];
        const float* src = &data_.x[static_cast<std::size_t>(s) * kInputDim];
        real* dst = X + static_cast<std::size_t>(i) * kInputDim;
        for (int j = 0; j < kInputDim; ++j) dst[j] = static_cast<real>(src[j]);
        y[i] = data_.y[static_cast<std::size_t>(s)];
    }
}

void SyntheticSource::fill(long step, long batch, long first, long rows, real* X, std::uint8_t* y) {
    Dataset d = make_synthetic(step * batch + first, rows, seed_);
    for (std::size_t i = 0; i < d.x.size(); ++i) X[i] = static_cast<real>(d.x[i]);
    std::copy(d.y.begin(), d.y.end(), y);
}

void evaluate(Model& model, const Dataset& data, const Comm& comm, double* loss, double* accuracy,
              EvalDetail* detail) {
    const std::size_t E = static_cast<std::size_t>(model.config().experts);
    if (detail) {
        detail->confusion.assign(kNumClasses * kNumClasses, 0.0);
        detail->routing.assign(model.moe_layers().size() * kNumClasses * E, 0.0);
    }
    const long lo = data.n * comm.rank / comm.size, hi = data.n * (comm.rank + 1) / comm.size;
    const long chunk = 1024;
    // with expert parallelism every forward is collective, so every rank runs
    // the same number of chunks (the last ones may be empty)
    const long max_slice = (data.n + comm.size - 1) / comm.size;
    const long n_chunks = (max_slice + chunk - 1) / chunk;
    rvec X(static_cast<std::size_t>(chunk) * kInputDim);
    double sums[3] = {0, 0, 0};  // loss, correct, rows
    for (long c = 0; c < n_chunks; ++c) {
        const long s = std::min(lo + c * chunk, hi);
        const long rows = std::min(chunk, hi - s);
        for (long i = 0; i < rows * kInputDim; ++i)
            X[static_cast<std::size_t>(i)] = static_cast<real>(data.x[static_cast<std::size_t>(s * kInputDim + i)]);
        StepStats st = model.evaluate(X.data(), data.y.data() + s, static_cast<int>(rows));
        if (detail) {
            const std::uint8_t* y = data.y.data() + s;
            const rvec& z = model.last_logits();
            for (long i = 0; i < rows; ++i) {
                const real* zi = &z[static_cast<std::size_t>(i) * kNumClasses];
                const long pred = std::max_element(zi, zi + kNumClasses) - zi;  // same argmax rule as the accuracy
                detail->confusion[static_cast<std::size_t>(y[i]) * kNumClasses + static_cast<std::size_t>(pred)] += 1;
            }
            const int K = model.config().topk;
            for (std::size_t l = 0; l < model.moe_layers().size(); ++l) {
                const std::vector<int>& choice = model.moe_layers()[l]->last_choice();
                for (long i = 0; i < rows; ++i)
                    for (int j = 0; j < K; ++j)
                        detail->routing[(l * kNumClasses + y[i]) * E +
                                        static_cast<std::size_t>(choice[static_cast<std::size_t>(i * K + j)])] += 1;
            }
        }
        sums[0] += st.loss_sum;
        sums[1] += static_cast<double>(st.correct);
        sums[2] += static_cast<double>(st.rows);
    }
    if (comm.sum_doubles) {
        comm.sum_doubles(sums, 3);
        if (detail) {
            comm.sum_doubles(detail->confusion.data(), static_cast<int>(detail->confusion.size()));
            if (!detail->routing.empty()) comm.sum_doubles(detail->routing.data(), static_cast<int>(detail->routing.size()));
        }
    }
    *loss = sums[0] / sums[2];
    *accuracy = sums[1] / sums[2];
}

namespace {

struct LoadSummary {
    double imbalance = 0, cv = 0, dropped = 0, aux = 0;
};

// MoE load statistics of the steps since `since` (one snapshot per layer):
// max/mean and CV of the per-expert load and aux averaged over steps and
// layers (aux summed over layers), dropped fraction over all ranks. Collective.
LoadSummary moe_summary(const Model& model, const Comm& comm, const std::vector<MoEStats>& since) {
    LoadSummary r;
    double sums[2] = {0, 0};  // assignments, dropped (per rank)
    const auto& layers = model.moe_layers();
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const MoEStats& s = layers[i]->stats();
        const MoEStats& b = since[i];
        const double steps = static_cast<double>(s.steps - b.steps);
        sums[0] += s.assignments - b.assignments;
        sums[1] += s.dropped - b.dropped;
        if (steps <= 0) continue;
        r.imbalance += (s.max_load - b.max_load) / (s.mean_load - b.mean_load);
        r.cv += (s.cv_load - b.cv_load) / steps;
        r.aux += (s.aux_loss - b.aux_loss) / steps;
    }
    if (comm.sum_doubles) comm.sum_doubles(sums, 2);
    const double blocks = static_cast<double>(std::max<std::size_t>(layers.size(), 1));
    r.imbalance /= blocks;
    r.cv /= blocks;
    r.dropped = sums[0] > 0 ? sums[1] / sums[0] : 0;
    return r;
}

std::vector<MoEStats> snapshot(const Model& model) {
    std::vector<MoEStats> v;
    for (const auto& m : model.moe_layers()) v.push_back(m->stats());
    return v;
}

}  // namespace

TrainResult train(Model& model, BatchSource& source, const Dataset* test, const TrainConfig& cfg,
                  const Comm& comm) {
    if (cfg.batch % comm.size != 0) throw std::runtime_error("global batch must be divisible by the number of ranks");
    const long local = cfg.batch / comm.size;
    const long first = local * comm.rank;
    const long per_epoch = source.steps_per_epoch(cfg.batch);
    const long total_steps = cfg.steps > 0 ? cfg.steps : per_epoch * cfg.epochs;
    if (total_steps <= 0) throw std::runtime_error("no training steps (batch larger than the dataset?)");

    auto make_optimizer = [&]() -> std::unique_ptr<Optimizer> {
        if (cfg.optim == "sgd") return std::make_unique<SGD>(cfg.lr);
        if (cfg.optim == "adam") return std::make_unique<Adam>(cfg.lr);
        throw std::runtime_error("unknown optimiser " + cfg.optim);
    };
    std::unique_ptr<Optimizer> opt = make_optimizer(), opt_experts = make_optimizer();

    ParamStore& ps = model.params();
    ParamStore& pe = model.expert_params();
    rvec X(static_cast<std::size_t>(local) * kInputDim);
    std::vector<std::uint8_t> y(static_cast<std::size_t>(local));
    const real grad_scale = static_cast<real>(1.0 / static_cast<double>(cfg.batch));

    TrainResult res;
    std::vector<double> step_times;
    double epoch_sums[3] = {0, 0, 0};  // loss, correct, rows
    const std::vector<MoEStats> start = snapshot(model);
    std::vector<MoEStats> epoch_start = start, log_start = start;

    for (long step = 0; step < total_steps; ++step) {
        const bool timed = step >= cfg.warmup;
        PhaseTimer t;

        t.start("data");
        source.fill(step, cfg.batch, first, local, X.data(), y.data());
        t.stop("data");

        if (comm.take_alltoall_time) comm.take_alltoall_time();  // drop exchanges of an evaluation pass
        t.start("compute");
        ps.zero_grad();
        pe.zero_grad();
        StepStats st = model.forward_backward(X.data(), y.data(), static_cast<int>(local), grad_scale, first, cfg.batch);
        t.stop("compute");
        if (comm.take_alltoall_time) {
            const auto [wait, a2a] = comm.take_alltoall_time();
            t.add("a2a_wait", wait);
            t.add("a2a", a2a);
            t.add("compute", -(wait + a2a));
        }

        if (comm.sum_gradients) {
            // the barrier separates load-imbalance waiting from the reduction itself
            t.start("wait");
            if (comm.barrier) comm.barrier();
            t.stop("wait");
            t.start("comm");
            if (comm.zero) {
                comm.zero_dense.reduce_scatter(ps.grad.data(), ps.size());
                if (pe.size() > 0) comm.zero_experts.reduce_scatter(pe.grad.data(), pe.size());
            } else {
                comm.sum_gradients(ps.grad.data(), ps.size());
                if (pe.size() > 0) {
                    if (comm.sum_expert_gradients) comm.sum_expert_gradients(pe.grad.data(), pe.size());
                    else comm.sum_gradients(pe.grad.data(), pe.size());
                }
            }
            t.stop("comm");
        }

        t.start("optim");
        if (comm.zero) {
            const auto [lo, hi] = comm.zero_dense.shard(ps.size());
            opt->step(ps.value.data(), ps.grad.data(), lo, hi);
            if (pe.size() > 0) {
                const auto [elo, ehi] = comm.zero_experts.shard(pe.size());
                opt_experts->step(pe.value.data(), pe.grad.data(), elo, ehi);
            }
        } else {
            opt->step(ps.value.data(), ps.grad.data(), 0, ps.size());
            if (pe.size() > 0) opt_experts->step(pe.value.data(), pe.grad.data(), 0, pe.size());
        }
        t.stop("optim");

        if (comm.zero) {
            t.start("comm");
            comm.zero_dense.all_gather(ps.value.data(), ps.size());
            if (pe.size() > 0) comm.zero_experts.all_gather(pe.value.data(), pe.size());
            t.stop("comm");
        }

        if (timed) {
            const double step_time =
                t.total("compute") + t.total("a2a_wait") + t.total("a2a") + t.total("wait") + t.total("comm") +
                t.total("optim");
            step_times.push_back(step_time);
            for (const auto& kv : t.totals()) res.phases.add(kv.first, kv.second);
        }

        // routing load over training (E6): one LOAD line every log_every steps
        if (model.is_moe() && cfg.log_every > 0 && (step + 1) % cfg.log_every == 0) {
            const LoadSummary l = moe_summary(model, comm, log_start);
            log_start = snapshot(model);
            if (comm.rank == 0 && !cfg.quiet)
                std::printf("LOAD exp=%s step=%ld imbalance=%.4f cv=%.4f dropped=%.5f aux=%.5f\n", cfg.exp.c_str(),
                            step + 1, l.imbalance, l.cv, l.dropped, l.aux);
        }

        epoch_sums[0] += st.loss_sum;
        epoch_sums[1] += static_cast<double>(st.correct);
        epoch_sums[2] += static_cast<double>(st.rows);
        const bool epoch_end = cfg.steps == 0 && (step + 1) % per_epoch == 0;
        if (epoch_end || step + 1 == total_steps) {
            if (comm.sum_doubles) comm.sum_doubles(epoch_sums, 3);
            res.final_train_loss = epoch_sums[0] / epoch_sums[2];
            double test_loss = 0, test_acc = -1;
            const bool last = step + 1 == total_steps;
            EvalDetail detail;
            if (test) {
                evaluate(model, *test, comm, &test_loss, &test_acc, last ? &detail : nullptr);
                res.test_accuracy = test_acc;
            }
            // E11 on the test set after training: confusion matrix (CONF, one line per
            // true class) and which experts each class is sent to (ROUTE)
            if (!detail.confusion.empty() && comm.rank == 0) {
                for (std::size_t c = 0; c < kNumClasses; ++c) {
                    std::printf("CONF exp=%s class=%zu counts=", cfg.exp.c_str(), c);
                    for (std::size_t q = 0; q < kNumClasses; ++q)
                        std::printf("%s%.0f", q ? ":" : "", detail.confusion[c * kNumClasses + q]);
                    std::printf("\n");
                }
                const std::size_t E = static_cast<std::size_t>(model.config().experts);
                const std::vector<double>& routing = detail.routing;
                for (std::size_t l = 0; l < model.moe_layers().size(); ++l)
                    for (std::size_t c = 0; c < kNumClasses; ++c) {
                        std::printf("ROUTE exp=%s layer=%zu class=%zu counts=", cfg.exp.c_str(), l, c);
                        for (std::size_t e = 0; e < E; ++e)
                            std::printf("%s%.0f", e ? ":" : "", routing[(l * kNumClasses + c) * E + e]);
                        std::printf("\n");
                    }
            }
            LoadSummary l;  // MoE load of this epoch
            if (model.is_moe()) {
                l = moe_summary(model, comm, epoch_start);
                epoch_start = snapshot(model);
            }
            if (comm.rank == 0 && !cfg.quiet)
                std::printf("EPOCH exp=%s epoch=%ld step=%ld train_loss=%.6f train_acc=%.4f test_loss=%.6f test_acc=%.4f "
                            "aux=%.5f imbalance=%.4f cv=%.4f dropped=%.5f\n",
                            cfg.exp.c_str(), step / std::max(per_epoch, 1L), step + 1, res.final_train_loss,
                            epoch_sums[1] / epoch_sums[2], test_loss, test_acc, l.aux, l.imbalance, l.cv, l.dropped);
            epoch_sums[0] = epoch_sums[1] = epoch_sums[2] = 0;
        }
    }

    if (model.is_moe()) {
        const LoadSummary l = moe_summary(model, comm, start);
        res.drop_fraction = l.dropped;
        res.imbalance = l.imbalance;
        res.cv_load = l.cv;
        res.aux_mean = l.aux;
    }
    res.optim_state_bytes = static_cast<double>(opt->state_bytes() + opt_experts->state_bytes());
    res.timed_steps = static_cast<long>(step_times.size());
    if (!step_times.empty()) {
        double sum = 0;
        for (double v : step_times) sum += v;
        res.t_step_mean = sum / static_cast<double>(step_times.size());
        std::vector<double> sorted = step_times;
        std::sort(sorted.begin(), sorted.end());
        res.t_step_median = sorted[sorted.size() / 2];
    }
    return res;
}
