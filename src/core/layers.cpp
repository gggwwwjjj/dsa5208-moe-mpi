#include "layers.hpp"

#include <cmath>

#include "../backend/backend.hpp"
#include "rng.hpp"

// ---------------------------------------------------------------- Linear ----

Linear::Linear(ParamStore& store, int in, int out)
    : store_(&store), in_(in), out_(out),
      w_off_(store.add(static_cast<std::size_t>(in) * out)), b_off_(store.add(static_cast<std::size_t>(out))) {}

void Linear::init(std::uint64_t seed, std::uint64_t layer_id, double scale) {
    // He-uniform for ReLU networks: U(-a, a) with a = scale * sqrt(6 / fan_in)
    const double a = scale * std::sqrt(6.0 / in_);
    real* w = W();
    const std::size_t n = static_cast<std::size_t>(in_) * out_;
    for (std::size_t i = 0; i < n; ++i) w[i] = static_cast<real>((2.0 * uniform01(seed, layer_id, i) - 1.0) * a);
    std::fill(b(), b() + out_, real(0));
}

void Linear::forward(const real* X, int rows, real* Y) const {
    backend::gemm(false, false, rows, out_, in_, real(1), X, in_, W(), out_, real(0), Y, out_);
    backend::add_bias(Y, b(), rows, out_);
}

void Linear::backward(const real* X, const real* dY, int rows, real* dX) {
    // dW += X^T dY   (in x out)
    backend::gemm(true, false, in_, out_, rows, real(1), X, in_, dY, out_, real(1), dW(), out_);
    // db += column sums of dY
    backend::colsum_accumulate(dY, db(), rows, out_);
    // dX = dY W^T    (rows x in)
    if (dX) backend::gemm(false, true, rows, in_, out_, real(1), dY, out_, W(), out_, real(0), dX, in_);
}
