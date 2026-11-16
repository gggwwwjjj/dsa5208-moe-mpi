// backend::gemm and backend::gemm_naive vs a plain host triple loop (so on the
// CUDA backend the GPU kernels are checked against the CPU), all four
// transpose combinations and shapes that are not multiples of any tile size
// (plan §6, first row).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/backend/backend.hpp"
#include "../src/core/memory.hpp"

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    const int shapes[][3] = {{1, 1, 1}, {7, 5, 3}, {33, 17, 65}, {64, 64, 64}, {130, 70, 257}, {300, 200, 520}};
    const double tol = sizeof(real) == 8 ? 1e-12 : 1e-4;
    int failures = 0;
    for (const auto& s : shapes) {
        const int M = s[0], N = s[1], K = s[2];
        for (int ta = 0; ta < 2; ++ta) {
            for (int tb = 0; tb < 2; ++tb) {
                // stored shapes: A is M x K (or K x M if transposed), B is K x N (or N x K)
                const int lda = ta ? M : K, ldb = tb ? K : N;
                rvec A(static_cast<std::size_t>(M) * K), B(static_cast<std::size_t>(K) * N);
                rvec C0(static_cast<std::size_t>(M) * N), C1;
                for (auto& v : A) v = static_cast<real>(dist(rng));
                for (auto& v : B) v = static_cast<real>(dist(rng));
                for (auto& v : C0) v = static_cast<real>(dist(rng));
                C1 = C0;
                rvec C2 = C0;
                // host reference: k in increasing order, like both backends
                std::vector<real> ref(C0.size());
                for (int i = 0; i < M; ++i)
                    for (int j = 0; j < N; ++j) {
                        real acc = 0;
                        for (int k = 0; k < K; ++k)
                            acc += (ta ? A[static_cast<std::size_t>(k) * lda + i] : A[static_cast<std::size_t>(i) * lda + k]) *
                                   (tb ? B[static_cast<std::size_t>(j) * ldb + k] : B[static_cast<std::size_t>(k) * ldb + j]);
                        ref[static_cast<std::size_t>(i) * N + j] = real(0.5) * acc + real(2) * C0[static_cast<std::size_t>(i) * N + j];
                    }
                backend::gemm_naive(ta, tb, M, N, K, real(0.5), A.data(), lda, B.data(), ldb, real(2), C2.data(), N);
                backend::gemm(ta, tb, M, N, K, real(0.5), A.data(), lda, B.data(), ldb, real(2), C1.data(), N);
                double worst = 0;
                for (std::size_t i = 0; i < C0.size(); ++i)
                    worst = std::fmax(worst, std::fmax(std::fabs(double(ref[i]) - double(C1[i])), std::fabs(double(ref[i]) - double(C2[i]))));
                if (worst > tol) {
                    std::printf("FAIL M=%d N=%d K=%d ta=%d tb=%d max_abs_diff=%.3e\n", M, N, K, ta, tb, worst);
                    ++failures;
                }
            }
        }
    }
    std::printf("test_gemm [%s, %s]: %s\n", backend::name(), sizeof(real) == 8 ? "double" : "float",
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
