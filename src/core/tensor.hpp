#pragma once
// Row-major dense matrix. Activations are (batch x features), weights are
// (in_features x out_features), so a Linear layer is Y = X W + b.
#include <algorithm>
#include <cstddef>
#include <vector>

#include "config.hpp"

struct Mat {
    int rows = 0;
    int cols = 0;
    std::vector<real> data;

    Mat() = default;
    Mat(int r, int c) : rows(r), cols(c), data(static_cast<std::size_t>(r) * c, real(0)) {}

    real* ptr() { return data.data(); }
    const real* ptr() const { return data.data(); }
    real& at(int i, int j) { return data[static_cast<std::size_t>(i) * cols + j]; }
    real at(int i, int j) const { return data[static_cast<std::size_t>(i) * cols + j]; }
    std::size_t size() const { return data.size(); }
    void zero() { std::fill(data.begin(), data.end(), real(0)); }
};
