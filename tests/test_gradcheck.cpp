// Finite-difference check of the full model gradient (plan §6, row 2):
// input Linear + ReLU, residual FFN blocks, output Linear, softmax CE.
// Central differences in double precision on a spread of parameters that
// covers every layer's weights and biases.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/core/data.hpp"
#include "../src/core/model.hpp"

namespace {

// Checks d(mean CE + aux)/d(theta) against central differences for every
// 37th parameter of both stores. Returns the number of failures.
int check(const char* name, ModelConfig cfg, int rows) {
    Model model(cfg, 7);
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    rvec X(static_cast<std::size_t>(rows) * kInputDim);
    std::vector<std::uint8_t> y(static_cast<std::size_t>(rows));
    for (auto& v : X) v = u(rng);
    for (int i = 0; i < rows; ++i) y[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i % kNumClasses);

    ParamStore* stores[2] = {&model.params(), &model.expert_params()};
    for (ParamStore* ps : stores) ps->zero_grad();
    model.forward_backward(X.data(), y.data(), rows, real(1.0 / rows), 0, rows);
    const rvec analytic[2] = {stores[0]->grad, stores[1]->grad};

    auto total_loss = [&] {
        StepStats st = model.training_loss(X.data(), y.data(), rows, 0, rows);
        return st.loss_sum / rows + st.aux;
    };
    const double eps = 1e-6;
    int checked = 0, failures = 0;
    double worst = 0;
    for (int s = 0; s < 2; ++s) {
        ParamStore& ps = *stores[s];
        for (std::size_t i = 0; i < ps.size(); i += 37) {
            const real saved = ps.value[i];
            ps.value[i] = saved + eps;
            const double up = total_loss();
            ps.value[i] = saved - eps;
            const double down = total_loss();
            ps.value[i] = saved;
            const double numeric = (up - down) / (2 * eps);
            const double a = analytic[s][i];
            const double rel = std::fabs(a - numeric) / std::fmax(std::fabs(a) + std::fabs(numeric), 1e-7);
            worst = std::fmax(worst, rel);
            ++checked;
            // central differences in double carry a rounding error of about
            // 1e-16 * |loss| / eps ~ 5e-10, so tiny gradients are judged absolutely
            if (rel > 1e-6 && std::fabs(a - numeric) > 1e-9) {
                ++failures;
                if (failures <= 5)
                    std::printf("  %s store %d param %zu: analytic %.10e numeric %.10e rel %.2e\n", name, s, i, a, numeric, rel);
            }
        }
    }
    std::printf("test_gradcheck %-22s %4d parameters, worst relative error %.2e: %s\n", name, checked, worst,
                failures == 0 ? "PASS" : "FAIL");
    return failures;
}

}  // namespace

int main() {
    static_assert(sizeof(real) == 8, "build the gradient check with REAL=double");
    ModelConfig dense;
    dense.hidden = 12;
    dense.ffn = 9;
    dense.blocks = 2;
    int failures = check("dense", dense, 6);

    ModelConfig moe = dense;
    moe.experts = 4;
    moe.topk = 2;
    moe.capacity_factor = 0.75;  // small enough that some assignments are dropped
    moe.aux_weight = 0.1;
    failures += check("moe top-2 (with drops)", moe, 8);
    moe.topk = 1;
    moe.capacity_factor = 1.25;
    failures += check("moe top-1", moe, 8);
    return failures == 0 ? 0 : 1;
}
