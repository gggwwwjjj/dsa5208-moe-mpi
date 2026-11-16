#include "data.hpp"

#include "rng.hpp"

#include <cstdio>
#include <stdexcept>

namespace {

std::vector<char> read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path + " (run `make data` first)");
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(static_cast<std::size_t>(bytes));
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::fclose(f);
        throw std::runtime_error("short read on " + path);
    }
    std::fclose(f);
    return buf;
}

}  // namespace

Dataset load_fashion_mnist(const std::string& dir, const std::string& split) {
    std::vector<char> img = read_file(dir + "/" + split + "_images.f32");
    std::vector<char> lab = read_file(dir + "/" + split + "_labels.u8");
    Dataset d;
    d.n = static_cast<long>(lab.size());
    if (img.size() != static_cast<std::size_t>(d.n) * kInputDim * sizeof(float))
        throw std::runtime_error("image/label count mismatch in " + dir);
    const float* src = reinterpret_cast<const float*>(img.data());
    d.x.assign(src, src + static_cast<std::size_t>(d.n) * kInputDim);
    d.y.assign(lab.begin(), lab.end());
    return d;
}

Dataset make_synthetic(long first, long count, std::uint64_t seed) {
    // teacher weights: stream id 0, sample ids are offset by 1
    std::vector<double> teacher(static_cast<std::size_t>(kInputDim) * kNumClasses);
    for (std::size_t k = 0; k < teacher.size(); ++k) teacher[k] = uniform01(seed, 0, k) - 0.5;

    Dataset d;
    d.n = count;
    d.x.resize(static_cast<std::size_t>(count) * kInputDim);
    d.y.resize(static_cast<std::size_t>(count));
    for (long s = 0; s < count; ++s) {
        std::uint64_t id = static_cast<std::uint64_t>(first + s) + 1;
        float* row = &d.x[static_cast<std::size_t>(s) * kInputDim];
        double logits[kNumClasses] = {0};
        for (int j = 0; j < kInputDim; ++j) {
            double v = uniform01(seed, id, static_cast<std::uint64_t>(j));
            row[j] = static_cast<float>(v - 0.5);
            // centred input: with v in [0,1) one class takes ~77% of the labels
            // and the router collapses onto two experts (plan §3.2)
            for (int c = 0; c < kNumClasses; ++c) logits[c] += (v - 0.5) * teacher[static_cast<std::size_t>(j) * kNumClasses + c];
        }
        int best = 0;
        for (int c = 1; c < kNumClasses; ++c)
            if (logits[c] > logits[best]) best = c;
        d.y[static_cast<std::size_t>(s)] = static_cast<std::uint8_t>(best);
    }
    return d;
}
