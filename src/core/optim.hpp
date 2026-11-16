#pragma once
// Optimisers over a range [begin, end) of a flat parameter buffer. The range
// lets ZeRO-1 update only the shard a rank owns (plan D9); without ZeRO the
// range is the whole buffer.
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "../backend/backend.hpp"
#include "config.hpp"
#include "memory.hpp"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(real* value, const real* grad, std::size_t begin, std::size_t end) = 0;
    virtual std::size_t state_bytes() const = 0;
};

// Plain SGD: used for the serial-vs-parallel equivalence tests (plan D12),
// because Adam's m / sqrt(v) turns ~1e-16 differences in tiny gradient
// components into lr-sized differences in the weights.
class SGD : public Optimizer {
public:
    explicit SGD(double lr) : lr_(lr) {}
    void step(real* value, const real* grad, std::size_t begin, std::size_t end) override {
        backend::sgd_step(value, grad, begin, end, static_cast<real>(lr_));
    }
    std::size_t state_bytes() const override { return 0; }

private:
    double lr_;
};

// Adam (Kingma & Ba 2015) with bias correction. State is allocated only for
// [begin, end) of the first call, so a ZeRO-1 shard holds 1/P of m and v.
class Adam : public Optimizer {
public:
    explicit Adam(double lr, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        : lr_(lr), b1_(beta1), b2_(beta2), eps_(eps) {}

    void step(real* value, const real* grad, std::size_t begin, std::size_t end) override {
        if (m_.empty()) {
            base_ = begin;
            m_.assign(end - begin, 0.0);
            v_.assign(end - begin, 0.0);
        }
        ++t_;
        const double c1 = 1.0 - std::pow(b1_, static_cast<double>(t_));
        const double c2 = 1.0 - std::pow(b2_, static_cast<double>(t_));
        // the moments of [begin, end) start at index begin - base_ (0: the range never moves)
        backend::adam_step(value, grad, m_.data() + (begin - base_), v_.data() + (begin - base_), begin, end, lr_, b1_,
                           b2_, eps_, c1, c2);
    }
    std::size_t state_bytes() const override { return (m_.size() + v_.size()) * sizeof(double); }

private:
    double lr_, b1_, b2_, eps_;
    long t_ = 0;
    std::size_t base_ = 0;
    svec<double> m_, v_;
};
