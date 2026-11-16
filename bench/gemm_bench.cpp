// E0 / E1: GEMM speed, naive triple loop vs blocked kernel, for square sizes
// and the current OMP_NUM_THREADS. Gives gamma (time per flop) for the cost
// model and shows the cache effect discussed in L12 p24-31.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../src/backend/backend.hpp"
#include "../src/core/memory.hpp"
#include "../src/core/args.hpp"
#include "../src/core/timer.hpp"

int main(int argc, char** argv) {
    Args args(argc, argv);
    const long max_n = args.integer("max-n", 2048);
    const long max_naive = args.integer("max-naive", 512);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    for (long n = 64; n <= max_n; n *= 2) {
        rvec A(static_cast<std::size_t>(n * n), real(0.5)), B(A), C(A);
        for (const std::string impl : {"naive", "blocked"}) {
            if (impl == "naive" && n > max_naive) continue;
            const double flops = 2.0 * static_cast<double>(n) * n * n;
            // repeat until at least 0.2 s, best of the repetitions
            double best = 1e30, spent = 0;
            int reps = 0;
            while (spent < 0.2 || reps < 3) {
                const double t0 = wall_seconds();
                if (impl == "naive")
                    backend::gemm_naive(false, false, n, n, n, real(1), A.data(), n, B.data(), n, real(0), C.data(), n);
                else
                    backend::gemm(false, false, n, n, n, real(1), A.data(), n, B.data(), n, real(0), C.data(), n);
                const double t = wall_seconds() - t0;
                best = std::min(best, t);
                spent += t;
                ++reps;
            }
            std::printf("RESULT exp=E0 kind=gemm impl=%s n=%ld threads=%d t=%.6e gflops=%.3f real=%s\n", impl.c_str(),
                        n, threads, best, flops / best / 1e9, sizeof(real) == 8 ? "double" : "float");
            std::fflush(stdout);
        }
    }
    return 0;
}
