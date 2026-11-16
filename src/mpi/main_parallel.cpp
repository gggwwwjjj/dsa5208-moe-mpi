// Parallel MoE-MLP training: dense layers data-parallel, experts
// expert-parallel (plan D7); MPI between ranks, OpenMP (or CUDA) inside a rank.
// Communication variants are chosen on the command line so every experiment
// uses one binary (plan D8, D9).
//
// Layout (D7): ranks g*ep .. g*ep+ep-1 form expert-parallel (EP) group g and
// split the E experts of every MoE layer between them; the D = P / ep groups
// hold replicas of the experts, and the D ranks with the same EP rank form an
// expert-data-parallel group that sums their gradients. --ep 1 = data
// parallelism only (every rank holds all experts).
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../backend/backend.hpp"
#include "../core/args.hpp"
#include "../core/options.hpp"
#include "../parallel/allreduce.hpp"
#include "../parallel/alltoall.hpp"

namespace {

// Rank 0 reads the files and broadcasts them (plan D13): 256 ranks reading the
// same file at once would dominate start-up on a shared file system.
Dataset load_and_broadcast(const std::string& dir, const std::string& split, int rank) {
    Dataset d;
    if (rank == 0) d = load_fashion_mnist(dir, split);
    long n = d.n;
    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    d.n = n;
    d.x.resize(static_cast<std::size_t>(n) * kInputDim);
    d.y.resize(static_cast<std::size_t>(n));
    MPI_Bcast(d.x.data(), static_cast<int>(d.x.size()), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(d.y.data(), static_cast<int>(d.y.size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    return d;
}

// Rank 0 writes the dense parameters and the experts of EP group 0 (world
// ranks 0 .. ep-1), reordered from per-rank stores into serial order. Called by
// every rank of EP group 0.
void dump_parallel(const std::string& path, Model& model, int ep, MPI_Comm ep_comm, int rank) {
    const rvec& mine = model.expert_params().value;
    const int n = static_cast<int>(mine.size());
    rvec gathered(rank == 0 ? mine.size() * static_cast<std::size_t>(ep) : 0);
    MPI_Gather(mine.data(), n, mpi_real(), gathered.data(), n, mpi_real(), 0, ep_comm);
    if (rank != 0) return;
    // each rank's store is [layer 0 experts, layer 1 experts, ...], same size per layer
    const std::size_t blocks = model.moe_layers().empty() ? 1 : model.moe_layers().size();
    const std::size_t per = mine.size() / blocks;
    rvec ordered;
    ordered.reserve(gathered.size());
    for (std::size_t l = 0; l < blocks; ++l)
        for (std::size_t r = 0; r < static_cast<std::size_t>(ep); ++r) {
            auto first = gathered.begin() + static_cast<std::ptrdiff_t>(r * mine.size() + l * per);
            ordered.insert(ordered.end(), first, first + static_cast<std::ptrdiff_t>(per));
        }
    dump_params(path, model.params().value, ordered);
}

}  // namespace

int main(int argc, char** argv) {
    // MPI is called only by the master thread, outside OpenMP regions (plan D11).
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 0;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED not available");
        // one GPU per rank on the CUDA backend: rank i on a node uses GPU i mod #GPUs
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        int local_rank = 0;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_free(&node_comm);
        backend::init(local_rank);
        Args args(argc, argv);
        RunOptions o = parse_options(args);
        const std::string allreduce_variant = args.str("allreduce", "mpi");
        const std::string alltoall_variant = args.str("alltoall", "mpi");
        // barrier before every token exchange, timed separately: splits load-imbalance
        // waiting from the exchange itself (time breakdown runs; adds ~log P latency)
        const bool a2a_wait = args.has("a2a-wait");
        int threads = 1;
#ifdef _OPENMP
        threads = omp_get_max_threads();
#endif
        // expert-parallel group size: by default min(P, E) (D7)
        const bool moe = o.model.experts > 1;
        const int ep = moe ? static_cast<int>(args.integer("ep", std::min(size, o.model.experts))) : 1;
        if (ep < 1 || size % ep != 0) throw std::runtime_error("--ep must divide the number of ranks");
        if (moe && o.model.experts % ep != 0) throw std::runtime_error("--ep must divide --experts");
        const int edp = size / ep;  // replicas of every expert (D)

        MPI_Comm ep_comm = MPI_COMM_NULL, edp_comm = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, rank / ep, rank, &ep_comm);   // EP group: consecutive ranks
        MPI_Comm_split(MPI_COMM_WORLD, rank % ep, rank, &edp_comm);  // same experts, one per EP group

        double a2a_seconds = 0, a2a_wait_seconds = 0;
        ExpertComm ecomm;
        ecomm.ep_size = ep;
        ecomm.ep_rank = rank % ep;
        if (ep > 1)
            ecomm.exchange = [&](const rvec& send, const std::vector<int>& send_counts, int row_len,
                                 rvec& recv, std::vector<int>& recv_counts, bool counts_known) {
                if (a2a_wait) {
                    const double tw = MPI_Wtime();
                    MPI_Barrier(ep_comm);
                    a2a_wait_seconds += MPI_Wtime() - tw;
                }
                const double t0 = MPI_Wtime();
                alltoall(alltoall_variant, send, send_counts, row_len, recv, recv_counts, counts_known, ep_comm);
                a2a_seconds += MPI_Wtime() - t0;
            };
        // MoE routing counts (aux loss) are summed over all ranks
        ecomm.sum_counts = [](double* v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); };
        // Capacity token groups (D16): each EP group holds the contiguous tokens
        // [g*B/D, (g+1)*B/D) and every expert replica limits its own group.
        // The serial run matches this with --groups D.
        o.model.groups = edp;
        // identical seed on every rank -> identical initial weights, no broadcast needed
        Model model(o.model, o.train.seed, ecomm);
        if (!o.load.empty()) load_params(o.load, model, rank % ep, ep);  // every rank reads its own part

        Dataset train_set, test_set;
        std::unique_ptr<BatchSource> source;
        double t_load = MPI_Wtime();
        if (o.data == "fashion") {
            train_set = load_and_broadcast(o.data_dir, "train", rank);
            if (o.eval) test_set = load_and_broadcast(o.data_dir, "test", rank);
            source = std::make_unique<DatasetSource>(train_set, o.train.seed);
        } else {
            source = std::make_unique<SyntheticSource>(o.train.seed);
        }
        t_load = MPI_Wtime() - t_load;
        if (rank == 0)
            std::printf("parallel | backend=%s real=%s ranks=%d threads=%d ep=%d expert_replicas=%d dense_params=%zu "
                        "expert_params_per_rank=%zu batch=%ld experts=%d topk=%d allreduce=%s alltoall=%s data=%s "
                        "load=%.3fs\n",
                        backend::name(), sizeof(real) == 8 ? "double" : "float", size, threads, ep, edp,
                        model.params().size(), model.expert_params().size(), o.train.batch, o.model.experts,
                        o.model.topk, allreduce_variant.c_str(), alltoall_variant.c_str(), o.data.c_str(), t_load);

        Comm comm;
        comm.rank = rank;
        comm.size = size;
        comm.sum_gradients = [&](real* buf, std::size_t n) { allreduce(allreduce_variant, buf, n, MPI_COMM_WORLD); };
        if (ep > 1) {
            // experts: only the replicas of the same experts are summed (nothing to do when D = 1)
            comm.sum_expert_gradients = [&](real* buf, std::size_t n) {
                if (edp > 1) allreduce(allreduce_variant, buf, n, edp_comm);
            };
            comm.take_alltoall_time = [&] {
                const std::pair<double, double> t{a2a_wait_seconds, a2a_seconds};
                a2a_wait_seconds = a2a_seconds = 0;
                return t;
            };
        }
        // ZeRO-1 (D9): dense shards over all ranks, expert shards over the ranks
        // holding the same experts (all ranks with --ep 1)
        comm.zero = args.has("zero");
        const MPI_Comm expert_dp = ep > 1 ? edp_comm : MPI_COMM_WORLD;
        auto sharded = [&](MPI_Comm c) {
            Comm::Sharded z;
            z.shard = [&, c](std::size_t n) {
                const Shard s = zero_shard(allreduce_variant, n, c);
                return std::make_pair(s.lo, s.hi);
            };
            z.reduce_scatter = [&, c](real* buf, std::size_t n) { reduce_scatter(allreduce_variant, buf, n, c); };
            z.all_gather = [&, c](real* buf, std::size_t n) { all_gather(allreduce_variant, buf, n, c); };
            return z;
        };
        comm.zero_dense = sharded(MPI_COMM_WORLD);
        comm.zero_experts = sharded(expert_dp);
        comm.sum_doubles = [](double* v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); };
        comm.barrier = [] { MPI_Barrier(MPI_COMM_WORLD); };

        TrainResult r = train(model, *source, o.eval ? &test_set : nullptr, o.train, comm);
        double t_max = r.t_step_mean;
        MPI_Allreduce(MPI_IN_PLACE, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            const std::string kind = (ep > 1 ? "ep_" + alltoall_variant + "_" + allreduce_variant : "dp_" + allreduce_variant) +
                                     (comm.zero ? "_zero1" : "");
            print_result(kind.c_str(), o, r, size, threads, ep, model.flops_per_sample(), t_max);
        }
        if (!o.dump.empty() && rank / ep == 0) dump_parallel(o.dump, model, ep, ep_comm, rank);
        MPI_Comm_free(&ep_comm);
        MPI_Comm_free(&edp_comm);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rank %d error: %s\n", rank, e.what());
        status = 1;
    }
    if (status != 0) MPI_Abort(MPI_COMM_WORLD, status);
    MPI_Finalize();
    return 0;
}
