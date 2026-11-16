#pragma once
// Compute backend interface (plan D9). The model and the parallel code only
// call these functions; cpu_omp.cpp implements them with OpenMP and
// cuda/*.cu with CUDA kernels. No BLAS or cuBLAS anywhere (L12 p16).
#include <cstddef>

#include "../core/config.hpp"

namespace backend {

// C = alpha * op(A) * op(B) + beta * C, all row-major.
// op(A) is M x K, op(B) is K x N, C is M x N.
// lda / ldb / ldc are the row strides of the stored (untransposed) arrays.
// Every C[i][j] is accumulated over k in increasing order, so the result does
// not depend on the number of threads.
void gemm(bool trans_a, bool trans_b, int M, int N, int K,
          real alpha, const real* A, int lda, const real* B, int ldb,
          real beta, real* C, int ldc);

// Reference triple loop used by the tests and by E1 (naive vs blocked).
void gemm_naive(bool trans_a, bool trans_b, int M, int N, int K,
                real alpha, const real* A, int lda, const real* B, int ldb,
                real beta, real* C, int ldc);

// Y[i][j] += b[j] for a rows x cols matrix.
void add_bias(real* Y, const real* b, int rows, int cols);
// db[j] += sum_i dY[i][j] (rows summed in increasing order).
void colsum_accumulate(const real* dY, real* db, int rows, int cols);
// Y = max(X, 0); X and Y may alias.
void relu_forward(const real* X, real* Y, std::size_t n);
// dX = dY where X > 0, else 0; dY and dX may alias.
void relu_backward(const real* X, const real* dY, real* dX, std::size_t n);
// Y += X
void add_inplace(real* Y, const real* X, std::size_t n);

// Optimiser updates over [begin, end) (optim.hpp keeps the hyper-parameters
// and the step count; ZeRO-1 passes only this rank's shard).
void sgd_step(real* value, const real* grad, std::size_t begin, std::size_t end, real lr);
// m, v hold the moments of [begin, end) (index i - begin); c1, c2 = bias corrections.
void adam_step(real* value, const real* grad, double* m, double* v, std::size_t begin, std::size_t end,
               double lr, double b1, double b2, double eps, double c1, double c2);

// Picks the device for this process (CUDA: GPU local_rank mod #GPUs; CPU: nothing).
void init(int local_rank);

// Memory that host code and the backend's kernels can both use (memory.hpp):
// malloc on the CPU backend, cudaMallocManaged on the CUDA backend.
void* alloc_shared(std::size_t bytes);
void free_shared(void* p) noexcept;

const char* name();

}  // namespace backend
