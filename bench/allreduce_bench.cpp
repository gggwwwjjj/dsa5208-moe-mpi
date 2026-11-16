// E5: all-reduce time vs message size, MPI_Allreduce vs our ring (plan E5,
// L8 p18-19, p40). Small messages are latency-bound (the library's tree /
// recursive doubling needs log2 P steps, the ring 2(P-1)); large ones are
// bandwidth-bound (the ring sends 2(P-1)/P n per rank). One RESULT line per
// size and variant; the time is the slowest rank's mean over the repetitions.
#include <mpi.h>

#include <cstdio>
#include <string>
#include <vector>

#include "../src/parallel/allreduce.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<real> buf((64u << 20) / sizeof(real), real(0));  // zeros: repeated sums cannot overflow
    for (long bytes = 8; bytes <= (64L << 20); bytes *= 4) {
        const std::size_t n = static_cast<std::size_t>(bytes) / sizeof(real);
        const int reps = bytes <= (1L << 16) ? 200 : (bytes <= (1L << 22) ? 20 : 5);
        for (const std::string variant : {"mpi", "ring"}) {
            allreduce(variant, buf.data(), n, MPI_COMM_WORLD);  // warm-up
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            for (int r = 0; r < reps; ++r) allreduce(variant, buf.data(), n, MPI_COMM_WORLD);
            double t = (MPI_Wtime() - t0) / reps;
            MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0)
                std::printf("RESULT exp=E5 kind=allreduce variant=%s ranks=%d bytes=%ld reps=%d t=%.6e\n",
                            variant.c_str(), size, bytes, reps, t);
        }
    }
    MPI_Finalize();
    return 0;
}
