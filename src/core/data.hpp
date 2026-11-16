#pragma once
// Datasets. Both sources give the same view: N samples of 784 features in
// [0, 1] and an integer label in [0, 10).
#include <cstdint>
#include <string>
#include <vector>

#include "config.hpp"

constexpr int kInputDim = 784;
constexpr int kNumClasses = 10;

struct Dataset {
    long n = 0;
    // Stored as float even in double builds: every rank keeps a full copy
    // (188 MB for Fashion-MNIST), and batches are converted to `real` on use.
    std::vector<float> x;         // n x kInputDim, row-major
    std::vector<std::uint8_t> y;  // n
};

// Fashion-MNIST converted by scripts/get_data.py:
//   <dir>/<split>_images.f32  (n x 784 float32, pixels scaled to [0, 1])
//   <dir>/<split>_labels.u8   (n uint8)
// split is "train" or "test".
Dataset load_fashion_mnist(const std::string& dir, const std::string& split);

// Synthetic data for timing runs (plan §3.2). Sample i depends only on
// (seed, i), so any rank can generate any subset without I/O and every rank
// count sees identical samples. Inputs are uniform in [-0.5, 0.5): zero-mean
// features, otherwise every token shares one large component after the input
// layer and the MoE router sends all tokens to the same experts (plan §3.2).
// The label is the argmax of a fixed random linear "teacher" 784 -> 10 drawn
// from the same seed (zero-mean input also keeps the 10 classes balanced).
Dataset make_synthetic(long first, long count, std::uint64_t seed);
