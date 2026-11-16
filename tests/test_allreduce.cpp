// Ring all-reduce and the ZeRO-1 halves (reduce-scatter + all-gather, both
// variants) vs MPI_Allreduce on random buffers, including sizes that
// are smaller than or not divisible by the number of ranks (plan §6).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/parallel/allreduce.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const double tol = sizeof(real) == 8 ? 1e-12 : 1e-4;
    int failures = 0;
    for (std::size_t n : {std::size_t(1), std::size_t(3), std::size_t(7), std::size_t(1000), std::size_t(12345)}) {
        std::mt19937 rng(1000 + static_cast<unsigned>(rank));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<real> a(n), b;
        for (auto& v : a) v = static_cast<real>(dist(rng));
        b = a;
        std::vector<real> zm = a, zr = a, ring = a;
        allreduce_mpi(a.data(), n, MPI_COMM_WORLD);
        allreduce_ring(b.data(), n, MPI_COMM_WORLD);
        double worst = 0;
        for (std::size_t i = 0; i < n; ++i) worst = std::fmax(worst, std::fabs(double(a[i]) - double(b[i])));
        // ZeRO-1 halves: reduce-scatter, then all-gather, must give the all-reduce;
        // the ring version must equal allreduce_ring bit for bit (same order of additions)
        bool same_as_ring = true, shards_cover = true;
        for (const char* v : {"mpi", "ring"}) {
            std::vector<real>& z = std::string(v) == "mpi" ? zm : zr;
            reduce_scatter(v, z.data(), n, MPI_COMM_WORLD);
            const Shard sh = zero_shard(v, n, MPI_COMM_WORLD);
            for (std::size_t i = sh.lo; i < sh.hi; ++i) worst = std::fmax(worst, std::fabs(double(z[i]) - double(a[i])));
            all_gather(v, z.data(), n, MPI_COMM_WORLD);
            for (std::size_t i = 0; i < n; ++i) worst = std::fmax(worst, std::fabs(double(z[i]) - double(a[i])));
            unsigned long owned = sh.hi - sh.lo;
            MPI_Allreduce(MPI_IN_PLACE, &owned, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            shards_cover = shards_cover && owned == n;
        }
        for (std::size_t i = 0; i < n; ++i) same_as_ring = same_as_ring && zr[i] == b[i];
        if (!same_as_ring || !shards_cover) worst = 1.0;
        MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (worst > tol) {
            ++failures;
            if (rank == 0) std::printf("  n=%zu max_abs_diff=%.3e\n", n, worst);
        }
    }
    if (rank == 0) std::printf("test_allreduce [%d ranks]: %s\n", size, failures == 0 ? "PASS" : "FAIL");
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
