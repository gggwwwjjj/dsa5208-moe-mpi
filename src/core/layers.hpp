#pragma once
// Flat parameter storage and the Linear layer, shared by the dense blocks and
// the MoE blocks.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.hpp"
#include "memory.hpp"

struct ParamStore {
    rvec value;
    rvec grad;

    // Reserve n parameters; returns their offset. Pointers are only taken
    // after all parameters are added (the vectors may move while growing).
    std::size_t add(std::size_t n) {
        std::size_t off = value.size();
        value.resize(off + n, real(0));
        grad.resize(off + n, real(0));
        return off;
    }
    std::size_t size() const { return value.size(); }
    void zero_grad() { std::fill(grad.begin(), grad.end(), real(0)); }
};

// Y = X W + b with W stored in x out (row-major), b of length out.
class Linear {
public:
    Linear(ParamStore& store, int in, int out);
    // He-uniform weights U(-a, a), a = scale * sqrt(6 / fan_in); zero bias
    void init(std::uint64_t seed, std::uint64_t layer_id, double scale = 1.0);
    void forward(const real* X, int rows, real* Y) const;
    // Accumulates dW += X^T dY and db += colsum(dY); writes dX = dY W^T if dX != nullptr.
    void backward(const real* X, const real* dY, int rows, real* dX);
    int in() const { return in_; }
    int out() const { return out_; }

private:
    real* W() const { return store_->value.data() + w_off_; }
    real* b() const { return store_->value.data() + b_off_; }
    real* dW() const { return store_->grad.data() + w_off_; }
    real* db() const { return store_->grad.data() + b_off_; }

    ParamStore* store_;
    int in_, out_;
    std::size_t w_off_, b_off_;
};
