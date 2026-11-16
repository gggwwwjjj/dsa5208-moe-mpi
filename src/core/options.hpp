#pragma once
// Command-line options shared by the serial and parallel executables, and the
// RESULT line that scripts/parse_logs.py collects.
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.hpp"
#include "model.hpp"
#include "train.hpp"

struct RunOptions {
    ModelConfig model;
    TrainConfig train;
    std::string data = "fashion";  // "fashion" or "synthetic"
    std::string data_dir = "data/fashion_mnist";
    std::string dump;              // write final parameters here (tests, checkpoints)
    std::string load;              // start from a --dump file instead of random weights (plan D17)
    bool eval = true;
};

inline RunOptions parse_options(const Args& a) {
    RunOptions o;
    o.model.hidden = static_cast<int>(a.integer("hidden", 512));
    o.model.ffn = static_cast<int>(a.integer("ffn", 1024));
    o.model.blocks = static_cast<int>(a.integer("blocks", 2));
    o.model.experts = static_cast<int>(a.integer("experts", 1));
    o.model.topk = static_cast<int>(a.integer("topk", 2));
    o.model.capacity_factor = a.number("capacity", 1.25);
    o.model.aux_weight = a.number("aux", 0.1);
    o.model.groups = static_cast<int>(a.integer("groups", 1));
    if (o.model.experts > 1 && o.model.topk > o.model.experts) throw std::runtime_error("--topk larger than --experts");
    o.train.batch = a.integer("batch", 256);
    o.train.steps = a.integer("steps", 0);
    o.train.epochs = static_cast<int>(a.integer("epochs", 1));
    o.train.warmup = static_cast<int>(a.integer("warmup", 5));
    o.train.optim = a.str("optim", "adam");
    o.train.lr = a.number("lr", o.train.optim == "sgd" ? 0.05 : 1e-3);
    o.train.seed = static_cast<std::uint64_t>(a.integer("seed", 42));
    o.train.exp = a.str("exp", "local");
    o.train.quiet = a.has("quiet");
    o.train.log_every = a.integer("log-every", 0);
    o.data = a.str("data", "fashion");
    o.data_dir = a.str("data-dir", "data/fashion_mnist");
    o.dump = a.str("dump", "");
    o.load = a.str("load", "");
    o.eval = !a.has("no-eval") && o.data == "fashion";
    if (o.data != "fashion" && o.data != "synthetic") throw std::runtime_error("--data must be fashion or synthetic");
    return o;
}

// Dense parameters followed by all experts in serial order (layer by layer,
// expert 0 .. E-1 within a layer).
inline void dump_params(const std::string& path, const rvec& dense, const rvec& experts) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::fwrite(dense.data(), sizeof(real), dense.size(), f);
    std::fwrite(experts.data(), sizeof(real), experts.size(), f);
    std::fclose(f);
}

// Reads a --dump file (dense parameters, then all experts in serial order) into
// a model whose expert store holds experts [ep_rank*E/ep, (ep_rank+1)*E/ep) of
// every layer (ep = 1: all experts). The file size must match the model, so a
// checkpoint of another configuration or precision is rejected (plan D17:
// timing runs start from the same partly trained router).
inline void load_params(const std::string& path, Model& model, int ep_rank, int ep) {
    rvec& dense = model.params().value;
    rvec& experts = model.expert_params().value;
    const std::size_t blocks = model.moe_layers().empty() ? 1 : model.moe_layers().size();
    const std::size_t local_per_layer = experts.size() / blocks;
    const std::size_t expected = dense.size() + experts.size() * static_cast<std::size_t>(ep);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + path);
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    if (bytes != static_cast<long>(expected * sizeof(real))) {
        std::fclose(f);
        throw std::runtime_error(path + ": size does not match this model and precision");
    }
    std::fseek(f, 0, SEEK_SET);
    std::size_t got = std::fread(dense.data(), sizeof(real), dense.size(), f);
    for (std::size_t l = 0; l < blocks; ++l) {
        // layer l of the file holds ep * local_per_layer values; this rank's are block ep_rank
        const std::size_t at = dense.size() + (l * static_cast<std::size_t>(ep) + static_cast<std::size_t>(ep_rank)) * local_per_layer;
        std::fseek(f, static_cast<long>(at * sizeof(real)), SEEK_SET);
        got += std::fread(experts.data() + l * local_per_layer, sizeof(real), local_per_layer, f);
    }
    std::fclose(f);
    if (got != dense.size() + experts.size()) throw std::runtime_error("short read from " + path);
}

// One machine-readable line per run (plan D14). Times are per timed step.
// ep: ranks per expert-parallel group (1 = experts replicated, data parallelism only).
inline void print_result(const char* kind, const RunOptions& o, const TrainResult& r, int ranks, int threads, int ep,
                         double flops_per_sample, double t_step_max) {
    const double steps = r.timed_steps > 0 ? static_cast<double>(r.timed_steps) : 1.0;
    std::printf("RESULT exp=%s kind=%s ranks=%d threads=%d ep=%d groups=%d data=%s batch=%ld hidden=%d ffn=%d blocks=%d experts=%d "
                "topk=%d capacity=%.2f optim=%s timed_steps=%ld t_step_mean=%.6e t_step_median=%.6e t_step_max=%.6e "
                "t_compute=%.6e t_a2a_wait=%.6e t_a2a=%.6e t_wait=%.6e t_comm=%.6e t_optim=%.6e t_data=%.6e gflops=%.3f "
                "train_loss=%.6f test_acc=%.4f aux=%.5f imbalance=%.4f cv_load=%.4f dropped=%.5f optim_state_mb=%.3f real=%s\n",
                o.train.exp.c_str(), kind, ranks, threads, ep, o.model.groups, o.data.c_str(), o.train.batch, o.model.hidden,
                o.model.ffn, o.model.blocks, o.model.experts, o.model.topk, o.model.capacity_factor, o.train.optim.c_str(), r.timed_steps, r.t_step_mean,
                r.t_step_median, t_step_max, r.phases.total("compute") / steps, r.phases.total("a2a_wait") / steps, r.phases.total("a2a") / steps, r.phases.total("wait") / steps,
                r.phases.total("comm") / steps, r.phases.total("optim") / steps, r.phases.total("data") / steps,
                r.t_step_mean > 0 ? flops_per_sample * static_cast<double>(o.train.batch) / r.t_step_mean / 1e9 : 0.0,
                r.final_train_loss, r.test_accuracy, r.aux_mean, r.imbalance, r.cv_load, r.drop_fraction, r.optim_state_bytes / 1e6,
                sizeof(real) == 8 ? "double" : "float");
}
