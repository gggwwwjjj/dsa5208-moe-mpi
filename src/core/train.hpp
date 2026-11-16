#pragma once
// One training loop for both executables (plan D4, D10). The serial build
// passes an empty Comm; the parallel build fills in MPI-based hooks. Keeping
// a single loop guarantees serial and parallel runs compute the same update.
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "data.hpp"
#include "model.hpp"
#include "timer.hpp"

// Where the samples of global batch `step` come from. Rank r of P gets rows
// [r*B/P, (r+1)*B/P) of the global batch, so every rank count sees the same
// global batches in the same order.
class BatchSource {
public:
    virtual ~BatchSource() = default;
    // Fills `rows` samples starting at position `first` of global batch `step`.
    virtual void fill(long step, long batch, long first, long rows, real* X, std::uint8_t* y) = 0;
    virtual long steps_per_epoch(long batch) const = 0;
};

// Fashion-MNIST: a fresh portable permutation per epoch, last partial batch dropped.
class DatasetSource : public BatchSource {
public:
    DatasetSource(const Dataset& data, std::uint64_t seed) : data_(data), seed_(seed) {}
    void fill(long step, long batch, long first, long rows, real* X, std::uint8_t* y) override;
    long steps_per_epoch(long batch) const override { return data_.n / batch; }

private:
    const Dataset& data_;
    std::uint64_t seed_;
    long perm_epoch_ = -1;
    std::vector<long> perm_;
};

// Synthetic data (plan §3.2): sample id of position j in batch `step` is
// step*B + j, generated on demand, so no rank stores the whole set.
class SyntheticSource : public BatchSource {
public:
    explicit SyntheticSource(std::uint64_t seed) : seed_(seed) {}
    void fill(long step, long batch, long first, long rows, real* X, std::uint8_t* y) override;
    long steps_per_epoch(long) const override { return 1L << 40; }

private:
    std::uint64_t seed_;
};

// Communication hooks; all empty for the serial build.
struct Comm {
    int rank = 0;
    int size = 1;
    std::function<void(real*, std::size_t)> sum_gradients;  // dense parameters: in-place sum over all ranks
    // expert parameters: sum over the ranks holding the same experts
    // (all ranks without expert parallelism, the expert-data-parallel group with it)
    std::function<void(real*, std::size_t)> sum_expert_gradients;
    std::function<void(double*, int)> sum_doubles;          // metrics
    std::function<void()> barrier;                          // separates waiting from communication
    // seconds spent in MoE token exchanges since the last call: {waiting at a
    // barrier before the exchanges (load imbalance; 0 without --a2a-wait),
    // the exchanges themselves}. They run inside forward/backward, so the loop
    // moves them from "compute" to "a2a_wait" and "a2a".
    std::function<std::pair<double, double>()> take_alltoall_time;

    // ZeRO-1 (plan D9, --zero): instead of summing whole gradients, each store
    // is reduce-scattered, the optimiser updates only this rank's shard
    // [lo, hi) (so it keeps Adam moments for that shard only), and the new
    // values are all-gathered. dense: over all ranks; experts: over the ranks
    // holding the same experts.
    struct Sharded {
        std::function<std::pair<std::size_t, std::size_t>(std::size_t n)> shard;
        std::function<void(real*, std::size_t)> reduce_scatter, all_gather;
    };
    bool zero = false;
    Sharded zero_dense, zero_experts;
};

struct TrainConfig {
    long batch = 256;        // global batch size B
    long steps = 0;          // > 0: run exactly this many steps (timing runs)
    int epochs = 1;          // used when steps == 0
    int warmup = 5;          // steps excluded from timing
    std::string optim = "adam";
    double lr = 1e-3;
    std::uint64_t seed = 42;
    std::string exp = "local";
    bool quiet = false;
    long log_every = 0;      // MoE: print a LOAD line every this many steps (0 = off)
};

struct TrainResult {
    double t_step_mean = 0;   // compute + wait + comm + optimiser, per timed step
    double t_step_median = 0;
    PhaseTimer phases;        // totals over timed steps
    long timed_steps = 0;
    double final_train_loss = 0;
    double test_accuracy = -1;
    // MoE (zero for dense models)
    double aux_mean = 0;        // auxiliary loss per step, summed over blocks
    double imbalance = 0;       // mean over steps and blocks of max/mean expert load
    double cv_load = 0;         // mean over steps and blocks of std/mean expert load
    double drop_fraction = 0;   // assignments dropped by the capacity limit
    double optim_state_bytes = 0;  // optimiser state held by this rank (Adam: 2 doubles per parameter it updates)
};

TrainResult train(Model& model, BatchSource& source, const Dataset* test, const TrainConfig& cfg,
                  const Comm& comm);

// E11 details of one evaluation, summed over all ranks.
struct EvalDetail {
    std::vector<double> confusion;  // [true class][predicted class]
    std::vector<double> routing;    // MoE: [layer][true class][expert], counts of top-k choices
};

// Mean loss and accuracy on `data`, rows split across ranks; fills `detail` if given.
void evaluate(Model& model, const Dataset& data, const Comm& comm, double* loss, double* accuracy,
              EvalDetail* detail = nullptr);
