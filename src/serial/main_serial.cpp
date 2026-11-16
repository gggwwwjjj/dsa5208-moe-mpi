// Serial MoE-MLP training (plan D10): no MPI, one thread. T_s for speedup and
// efficiency comes from this executable.
#include <cstdio>
#include <exception>
#include <memory>

#include "../backend/backend.hpp"
#include "../core/args.hpp"
#include "../core/options.hpp"

int main(int argc, char** argv) {
    try {
        RunOptions o = parse_options(Args(argc, argv));
        backend::init(0);
        Model model(o.model, o.train.seed);
        if (!o.load.empty()) load_params(o.load, model, 0, 1);

        Dataset train_set, test_set;
        std::unique_ptr<BatchSource> source;
        if (o.data == "fashion") {
            train_set = load_fashion_mnist(o.data_dir, "train");
            if (o.eval) test_set = load_fashion_mnist(o.data_dir, "test");
            source = std::make_unique<DatasetSource>(train_set, o.train.seed);
        } else {
            source = std::make_unique<SyntheticSource>(o.train.seed);
        }
        std::printf("serial | backend=%s real=%s dense_params=%zu expert_params=%zu batch=%ld hidden=%d ffn=%d "
                    "blocks=%d experts=%d topk=%d groups=%d data=%s\n",
                    backend::name(), sizeof(real) == 8 ? "double" : "float", model.params().size(),
                    model.expert_params().size(), o.train.batch, o.model.hidden, o.model.ffn, o.model.blocks,
                    o.model.experts, o.model.topk, o.model.groups, o.data.c_str());

        Comm comm;  // serial: no communication hooks
        TrainResult r = train(model, *source, o.eval ? &test_set : nullptr, o.train, comm);
        print_result("serial", o, r, 1, 1, 1, model.flops_per_sample(), r.t_step_mean);
        if (!o.dump.empty()) dump_params(o.dump, model.params().value, model.expert_params().value);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
