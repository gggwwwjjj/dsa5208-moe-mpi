#pragma once
// MoE-MLP (plan D1):
//   x (784) -> Linear 784->H, ReLU -> [block] x blocks -> Linear H->10 -> softmax CE
// With --experts 1 each block is a dense residual FFN, h <- h + FFN(h),
// FFN(h) = Linear F->H (ReLU (Linear H->F (h))): the dense baseline.
// With --experts E > 1 each block is a MoE block (moe.hpp).
//
// Parameters live in two flat stores: `dense` (input, output and router
// layers, replicated on every data-parallel rank) and `experts` (only the
// experts this rank owns). Gradient all-reduce, optimiser updates and ZeRO
// sharding each work on one contiguous buffer.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "layers.hpp"
#include "moe.hpp"

struct ModelConfig {
    int input = 784;
    int hidden = 512;
    int ffn = 1024;
    int blocks = 2;
    int classes = 10;
    int experts = 1;               // 1 = dense FFN blocks
    int topk = 2;
    double capacity_factor = 1.25;
    double aux_weight = 0.1;
    int groups = 1;                // capacity token groups (plan D16)
};

struct StepStats {
    double loss_sum = 0;  // cross-entropy summed over the rows of this call (not averaged)
    double aux = 0;       // MoE auxiliary loss of this step (global value, summed over blocks)
    long correct = 0;
    long rows = 0;
};

class Model {
public:
    Model(const ModelConfig& cfg, std::uint64_t seed, const ExpertComm& comm = ExpertComm());
    // layers keep pointers into the stores and to comm_, so the model must stay where it was built
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Forward + backward on `rows` samples whose global batch positions start
    // at pos_offset. Gradients are ACCUMULATED and scaled by grad_scale
    // (1 / global batch, so the sum over data-parallel ranks is the batch mean).
    StepStats forward_backward(const real* X, const std::uint8_t* y, int rows, real grad_scale, long pos_offset,
                               long batch);
    // Training-mode forward only (capacity + aux loss), for the gradient check.
    StepStats training_loss(const real* X, const std::uint8_t* y, int rows, long pos_offset, long batch);
    // Evaluation: no capacity limit, no aux loss.
    StepStats evaluate(const real* X, const std::uint8_t* y, int rows);
    // Logits of the last forward pass, rows x classes (E11 confusion matrix).
    const rvec& last_logits() const { return logits_; }

    ParamStore& params() { return store_; }
    ParamStore& expert_params() { return expert_store_; }
    const ModelConfig& config() const { return cfg_; }
    bool is_moe() const { return !moe_.empty(); }
    const std::vector<std::unique_ptr<MoELayer>>& moe_layers() const { return moe_; }
    // Floating-point operations of one forward+backward per sample (for E1).
    double flops_per_sample() const;

private:
    void forward(const real* X, int rows, long pos_offset, long batch, bool train);
    StepStats loss(const std::uint8_t* y, int rows, real grad_scale, bool want_grad);

    ModelConfig cfg_;
    ParamStore store_;
    ParamStore expert_store_;
    ExpertComm comm_;
    Linear in_;
    std::vector<Linear> up_;    // dense blocks: H -> F
    std::vector<Linear> down_;  // dense blocks: F -> H
    std::vector<std::unique_ptr<MoELayer>> moe_;
    Linear out_;

    // activations of the last forward pass, kept for backward
    rvec a_in_;               // pre-ReLU of the input layer, rows x H
    std::vector<rvec> h_;     // h_[l] = input of block l (h_[blocks] = final), rows x H
    std::vector<rvec> u_;     // dense blocks: pre-ReLU, rows x F
    std::vector<rvec> r_;     // dense blocks: ReLU output, rows x F
    rvec logits_;             // rows x classes
    // gradient scratch
    rvec dh_, dr_, dlogits_, dtmp_;
};
