// CUDA backend (plan §5.5, Stage C): the same interface as cpu_omp.cpp, with
// our own kernels - no cuBLAS (L12 p16). Arrays live in unified memory
// (alloc_shared = cudaMallocManaged), so host code (MoE routing, loss, MPI)
// and kernels use the same pointers and the driver migrates pages on demand.
// Every call ends with cudaDeviceSynchronize: the host may read the result
// right after the call returns.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <new>

#include "../backend.hpp"

namespace backend {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

void finish(const char* what) {
    check(cudaGetLastError(), what);
    check(cudaDeviceSynchronize(), what);
}

constexpr int kThreads = 256;

inline int blocks_for(std::size_t n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

// ---------------------------------------------------------------- GEMM ----
// C = alpha op(A) op(B) + beta C. Each block computes a TM x TN tile of C with
// 16 x 16 threads; each thread keeps a 4 x 4 micro-tile in registers (register
// tiling, L10). The K dimension is walked in slices of TK: the block copies the
// TM x TK slice of op(A) and the TK x TN slice of op(B) into shared memory
// (coalesced: consecutive threads read consecutive addresses for either
// layout), then every thread does 4 x 4 x TK multiply-adds from shared memory.
// Each C[i][j] is accumulated over k in increasing order, like the CPU
// backend, so results do not depend on the launch configuration.
constexpr int TM = 64, TN = 64, TK = 16, RT = 4;  // RT x RT outputs per thread

__global__ void gemm_tiled(bool ta, bool tb, int M, int N, int K, real alpha, const real* __restrict__ A, int lda,
                           const real* __restrict__ B, int ldb, real beta, real* __restrict__ C, int ldc) {
    __shared__ real As[TK][TM + 1];  // As[k][m] = op(A)(m0 + m, k0 + k); +1 against bank conflicts
    __shared__ real Bs[TK][TN + 1];  // Bs[k][n] = op(B)(k0 + k, n0 + n)
    const int tx = threadIdx.x, ty = threadIdx.y, t = ty * 16 + tx;
    const int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN;
    real acc[RT][RT] = {};
    for (int k0 = 0; k0 < K; k0 += TK) {
        // 1024 values of each slice, 4 per thread
        for (int p = 0; p < TM * TK / 256; ++p) {
            const int e = t + 256 * p;
            // op(A) is M x K: stored row-major (contiguous in k) or transposed (contiguous in m)
            const int m = ta ? e % TM : e / TK, k = ta ? e / TM : e % TK;
            const int gm = m0 + m, gk = k0 + k;
            real a = 0;
            if (gm < M && gk < K) a = ta ? A[static_cast<std::size_t>(gk) * lda + gm] : A[static_cast<std::size_t>(gm) * lda + gk];
            As[k][m] = a;
            // op(B) is K x N: stored row-major (contiguous in n) or transposed (contiguous in k)
            const int n = tb ? e / TK : e % TN, kb = tb ? e % TK : e / TN;
            const int gn = n0 + n, gkb = k0 + kb;
            real b = 0;
            if (gn < N && gkb < K) b = tb ? B[static_cast<std::size_t>(gn) * ldb + gkb] : B[static_cast<std::size_t>(gkb) * ldb + gn];
            Bs[kb][n] = b;
        }
        __syncthreads();
        const int kmax = min(TK, K - k0);
        for (int k = 0; k < kmax; ++k) {
            real a[RT], b[RT];
            for (int i = 0; i < RT; ++i) a[i] = As[k][ty + 16 * i];
            for (int j = 0; j < RT; ++j) b[j] = Bs[k][tx + 16 * j];
            for (int i = 0; i < RT; ++i)
                for (int j = 0; j < RT; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    // threads with consecutive tx write consecutive columns (coalesced)
    for (int i = 0; i < RT; ++i) {
        const int gm = m0 + ty + 16 * i;
        if (gm >= M) continue;
        for (int j = 0; j < RT; ++j) {
            const int gn = n0 + tx + 16 * j;
            if (gn >= N) continue;
            real& c = C[static_cast<std::size_t>(gm) * ldc + gn];
            c = alpha * acc[i][j] + (beta == real(0) ? real(0) : beta * c);
        }
    }
}

// one thread per C element, k in increasing order (reference for the tests and E9)
__global__ void gemm_simple(bool ta, bool tb, int M, int N, int K, real alpha, const real* A, int lda, const real* B,
                            int ldb, real beta, real* C, int ldc) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) return;
    real acc = 0;
    for (int k = 0; k < K; ++k) {
        const real a = ta ? A[static_cast<std::size_t>(k) * lda + i] : A[static_cast<std::size_t>(i) * lda + k];
        const real b = tb ? B[static_cast<std::size_t>(j) * ldb + k] : B[static_cast<std::size_t>(k) * ldb + j];
        acc += a * b;
    }
    real& c = C[static_cast<std::size_t>(i) * ldc + j];
    c = alpha * acc + (beta == real(0) ? real(0) : beta * c);
}

// ------------------------------------------------------- element-wise ----
__global__ void k_add_bias(real* Y, const real* b, std::size_t n, int cols) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) Y[i] += b[i % cols];
}

// one thread per column, rows summed in increasing order (deterministic, same
// order as the CPU backend); neighbouring threads read neighbouring columns
__global__ void k_colsum(const real* dY, real* db, int rows, int cols) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    real s = 0;
    for (int i = 0; i < rows; ++i) s += dY[static_cast<std::size_t>(i) * cols + j];
    db[j] += s;
}

__global__ void k_relu(const real* X, real* Y, std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) Y[i] = X[i] > real(0) ? X[i] : real(0);
}

__global__ void k_relu_backward(const real* X, const real* dY, real* dX, std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) dX[i] = X[i] > real(0) ? dY[i] : real(0);
}

__global__ void k_add(real* Y, const real* X, std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) Y[i] += X[i];
}

__global__ void k_sgd(real* value, const real* grad, std::size_t n, real lr) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) value[i] -= lr * grad[i];
}

__global__ void k_adam(real* value, const real* grad, double* m, double* v, std::size_t n, double lr, double b1,
                       double b2, double eps, double c1, double c2) {
    const std::size_t s = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (s >= n) return;
    const double g = grad[s];
    m[s] = b1 * m[s] + (1.0 - b1) * g;
    v[s] = b2 * v[s] + (1.0 - b2) * g * g;
    value[s] -= static_cast<real>(lr * (m[s] / c1) / (sqrt(v[s] / c2) + eps));
}

}  // namespace

void gemm(bool trans_a, bool trans_b, int M, int N, int K, real alpha, const real* A, int lda, const real* B, int ldb,
          real beta, real* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    const dim3 threads(16, 16), grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    gemm_tiled<<<grid, threads>>>(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    finish("gemm");
}

void gemm_naive(bool trans_a, bool trans_b, int M, int N, int K, real alpha, const real* A, int lda, const real* B,
                int ldb, real beta, real* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    const dim3 threads(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    gemm_simple<<<grid, threads>>>(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    finish("gemm_naive");
}

void add_bias(real* Y, const real* b, int rows, int cols) {
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    if (n == 0) return;
    k_add_bias<<<blocks_for(n), kThreads>>>(Y, b, n, cols);
    finish("add_bias");
}

void colsum_accumulate(const real* dY, real* db, int rows, int cols) {
    if (cols <= 0) return;
    k_colsum<<<blocks_for(static_cast<std::size_t>(cols)), kThreads>>>(dY, db, rows, cols);
    finish("colsum");
}

void relu_forward(const real* X, real* Y, std::size_t n) {
    if (n == 0) return;
    k_relu<<<blocks_for(n), kThreads>>>(X, Y, n);
    finish("relu");
}

void relu_backward(const real* X, const real* dY, real* dX, std::size_t n) {
    if (n == 0) return;
    k_relu_backward<<<blocks_for(n), kThreads>>>(X, dY, dX, n);
    finish("relu_backward");
}

void add_inplace(real* Y, const real* X, std::size_t n) {
    if (n == 0) return;
    k_add<<<blocks_for(n), kThreads>>>(Y, X, n);
    finish("add");
}

void sgd_step(real* value, const real* grad, std::size_t begin, std::size_t end, real lr) {
    if (end <= begin) return;
    k_sgd<<<blocks_for(end - begin), kThreads>>>(value + begin, grad + begin, end - begin, lr);
    finish("sgd");
}

void adam_step(real* value, const real* grad, double* m, double* v, std::size_t begin, std::size_t end, double lr,
               double b1, double b2, double eps, double c1, double c2) {
    if (end <= begin) return;
    k_adam<<<blocks_for(end - begin), kThreads>>>(value + begin, grad + begin, m, v, end - begin, lr, b1, b2, eps, c1, c2);
    finish("adam");
}

void init(int local_rank) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        std::abort();
    }
    check(cudaSetDevice(local_rank % count), "cudaSetDevice");
}

void* alloc_shared(std::size_t bytes) {
    void* p = nullptr;
    if (cudaMallocManaged(&p, bytes > 0 ? bytes : 1) != cudaSuccess || !p) throw std::bad_alloc();
    return p;
}

void free_shared(void* p) noexcept { cudaFree(p); }

const char* name() { return "cuda"; }

}  // namespace backend
