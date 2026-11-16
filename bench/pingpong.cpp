// E0: point-to-point latency (alpha) and bandwidth (beta) between rank 0 and
// rank P-1. Placing the two ranks on different nodes (PBS) measures the
// network; on one node it measures shared memory. Prints one RESULT line per
// message size for scripts/parse_logs.py.
// --concurrent: all P ranks ping-pong at the same time, rank r with r + P/2
// (with P/2 ranks per node on two nodes every pair crosses the network), as in
// a collective where every rank sends at once. Reports the slowest pair's
// one-way time, the bandwidth per pair and the total over the P/2 pairs.
#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        if (rank == 0) std::fprintf(stderr, "pingpong needs at least 2 ranks\n");
        MPI_Finalize();
        return 1;
    }
    const bool concurrent = argc > 1 && std::strcmp(argv[1], "--concurrent") == 0;
    if (concurrent && size % 2 != 0) {
        if (rank == 0) std::fprintf(stderr, "--concurrent needs an even number of ranks\n");
        MPI_Finalize();
        return 1;
    }
    const int half = size / 2;
    const int peer = concurrent ? (rank < half ? rank + half : rank - half) : (rank == 0 ? size - 1 : 0);
    const bool active = concurrent || rank == 0 || rank == size - 1;
    const bool starts = concurrent ? rank < half : rank == 0;
    char host[MPI_MAX_PROCESSOR_NAME];
    int len = 0;
    MPI_Get_processor_name(host, &len);
    if (active && (!concurrent || rank == 0 || rank == size - 1)) std::printf("rank %d on %s\n", rank, host);
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<char> buf(64u << 20);
    for (long bytes = 8; bytes <= (64L << 20); bytes *= 4) {
        const int reps = bytes <= (1L << 16) ? 1000 : (bytes <= (1L << 22) ? 100 : 20);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        if (active) {
            for (int r = 0; r < reps; ++r) {
                if (starts) {
                    MPI_Send(buf.data(), static_cast<int>(bytes), MPI_CHAR, peer, 0, MPI_COMM_WORLD);
                    MPI_Recv(buf.data(), static_cast<int>(bytes), MPI_CHAR, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buf.data(), static_cast<int>(bytes), MPI_CHAR, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buf.data(), static_cast<int>(bytes), MPI_CHAR, peer, 0, MPI_COMM_WORLD);
                }
            }
        }
        double one_way = (MPI_Wtime() - t0) / (2.0 * reps), slowest = one_way;
        if (concurrent) MPI_Reduce(&one_way, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && !concurrent)
            std::printf("RESULT exp=E0 kind=pingpong ranks=%d bytes=%ld t_oneway=%.9e bw_GBps=%.4f\n",
                        size, bytes, one_way, bytes / one_way / 1e9);
        if (rank == 0 && concurrent)
            std::printf("RESULT exp=E0 kind=pingpong_concurrent ranks=%d pairs=%d bytes=%ld t_oneway=%.9e "
                        "bw_GBps=%.4f total_GBps=%.4f\n",
                        size, half, bytes, slowest, bytes / slowest / 1e9, half * bytes / slowest / 1e9);
    }
    MPI_Finalize();
    return 0;
}
