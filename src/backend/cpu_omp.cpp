// CPU backend. Built with OpenMP for the parallel version and without it for
// the serial version (the pragmas are then ignored), so both use the same code.
#include "backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>

namespace backend {

namespace {

// Tile sizes for the blocked GEMM (L9 p33-34, L12 p29-30). One tile of the
// accumulator (BM x BN) plus a BM x BK slice of A and a BK x BN slice of B is
// 64*256 + 64*256 + 256*256 values = 384 KB in fp32, sized for a per-core L2.
// Tuned with bench/gemm_bench (E0); see the report for the measured effect.
constexpr int BM = 64;
constexpr int BN = 256;
constexpr int BK = 256;

inline real load(const real* M, int ld, bool trans, int i, int j) {
    // element (i, j) of op(M)
    return trans ? M[static_cast<std::size_t>(j) * ld + i] : M[static_cast<std::size_t>(i) * ld + j];
}

// Row-major copy of op(M) for a transposed operand: rows x cols, stride cols.
std::vector<real> materialise_transpose(const real* M, int ld, int rows, int cols) {
    std::vector<real> out(static_cast<std::size_t>(rows) * cols);
#pragma omp parallel for schedule(static) if (static_cast<long>(rows) * cols > 65536)
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[static_cast<std::size_t>(i) * cols + j] = M[static_cast<std::size_t>(j) * ld + i];
    return out;
}

}  // namespace

void gemm_naive(bool trans_a, bool trans_b, int M, int N, int K,
                real alpha, const real* A, int lda, const real* B, int ldb,
                real beta, real* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            real acc = 0;
            for (int k = 0; k < K; ++k) acc += load(A, lda, trans_a, i, k) * load(B, ldb, trans_b, k, j);
            real& c = C[static_cast<std::size_t>(i) * ldc + j];
            c = alpha * acc + (beta == real(0) ? real(0) : beta * c);
        }
    }
}

void gemm(bool trans_a, bool trans_b, int M, int N, int K,
          real alpha, const real* A, int lda, const real* B, int ldb,
          real beta, real* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    // Transposed operands are copied once (O(MK + KN)) so the O(MNK) inner
    // loop always streams contiguous rows.
    std::vector<real> At, Bt;
    if (trans_a) {
        At = materialise_transpose(A, lda, M, K);
        A = At.data();
        lda = K;
    }
    if (trans_b) {
        Bt = materialise_transpose(B, ldb, K, N);
        B = Bt.data();
        ldb = N;
    }
    [[maybe_unused]] const long work = static_cast<long>(M) * N * std::max(K, 1);  // unused without OpenMP
#pragma omp parallel if (work > (1L << 16))
    {
        std::vector<real> acc(static_cast<std::size_t>(BM) * BN);
        // Each thread owns whole output tiles, so no two threads write the
        // same C element and no reduction across threads is needed.
#pragma omp for collapse(2) schedule(static)
        for (int i0 = 0; i0 < M; i0 += BM) {
            for (int j0 = 0; j0 < N; j0 += BN) {
                const int i1 = std::min(i0 + BM, M);
                const int j1 = std::min(j0 + BN, N);
                const int nj = j1 - j0;
                std::fill(acc.begin(), acc.end(), real(0));
                for (int k0 = 0; k0 < K; k0 += BK) {
                    const int k1 = std::min(k0 + BK, K);
                    for (int i = i0; i < i1; ++i) {
                        real* acc_row = &acc[static_cast<std::size_t>(i - i0) * BN];
                        const real* a_row = A + static_cast<std::size_t>(i) * lda;
                        for (int k = k0; k < k1; ++k) {
                            const real a = a_row[k];
                            const real* b_row = B + static_cast<std::size_t>(k) * ldb + j0;
                            for (int j = 0; j < nj; ++j) acc_row[j] += a * b_row[j];  // vectorised over j
                        }
                    }
                }
                for (int i = i0; i < i1; ++i) {
                    real* c_row = C + static_cast<std::size_t>(i) * ldc + j0;
                    const real* acc_row = &acc[static_cast<std::size_t>(i - i0) * BN];
                    if (beta == real(0)) {
                        for (int j = 0; j < nj; ++j) c_row[j] = alpha * acc_row[j];
                    } else {
                        for (int j = 0; j < nj; ++j) c_row[j] = alpha * acc_row[j] + beta * c_row[j];
                    }
                }
            }
        }
    }
}

void add_bias(real* Y, const real* b, int rows, int cols) {
#pragma omp parallel for schedule(static) if (static_cast<long>(rows) * cols > 65536)
    for (int i = 0; i < rows; ++i) {
        real* y = Y + static_cast<std::size_t>(i) * cols;
        for (int j = 0; j < cols; ++j) y[j] += b[j];
    }
}

void colsum_accumulate(const real* dY, real* db, int rows, int cols) {
    // parallel over columns; each column sums its rows in increasing order
#pragma omp parallel for schedule(static) if (static_cast<long>(rows) * cols > 65536)
    for (int j = 0; j < cols; ++j) {
        real s = 0;
        for (int i = 0; i < rows; ++i) s += dY[static_cast<std::size_t>(i) * cols + j];
        db[j] += s;
    }
}

void relu_forward(const real* X, real* Y, std::size_t n) {
#pragma omp parallel for schedule(static) if (n > 65536)
    for (std::size_t i = 0; i < n; ++i) Y[i] = X[i] > real(0) ? X[i] : real(0);
}

void relu_backward(const real* X, const real* dY, real* dX, std::size_t n) {
#pragma omp parallel for schedule(static) if (n > 65536)
    for (std::size_t i = 0; i < n; ++i) dX[i] = X[i] > real(0) ? dY[i] : real(0);
}

void add_inplace(real* Y, const real* X, std::size_t n) {
#pragma omp parallel for schedule(static) if (n > 65536)
    for (std::size_t i = 0; i < n; ++i) Y[i] += X[i];
}

void sgd_step(real* value, const real* grad, std::size_t begin, std::size_t end, real lr) {
#pragma omp parallel for schedule(static) if (end - begin > 65536)
    for (std::size_t i = begin; i < end; ++i) value[i] -= lr * grad[i];
}

void adam_step(real* value, const real* grad, double* m, double* v, std::size_t begin, std::size_t end,
               double lr, double b1, double b2, double eps, double c1, double c2) {
#pragma omp parallel for schedule(static) if (end - begin > 65536)
    for (std::size_t i = begin; i < end; ++i) {
        const std::size_t s = i - begin;
        const double g = grad[i];
        m[s] = b1 * m[s] + (1.0 - b1) * g;
        v[s] = b2 * v[s] + (1.0 - b2) * g * g;
        value[i] -= static_cast<real>(lr * (m[s] / c1) / (std::sqrt(v[s] / c2) + eps));
    }
}

void init(int) {}

void* alloc_shared(std::size_t bytes) {
    void* p = std::malloc(bytes > 0 ? bytes : 1);
    if (!p) throw std::bad_alloc();
    return p;
}

void free_shared(void* p) noexcept { std::free(p); }

const char* name() {
#ifdef _OPENMP
    return "cpu_omp";
#else
    return "cpu_serial";
#endif
}

}  // namespace backend
