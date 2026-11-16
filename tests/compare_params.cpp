// compare_params A B TOL: max absolute difference between two parameter dumps
// (raw arrays of `real`, written with --dump). Exit 1 if above TOL.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/core/config.hpp"

static std::vector<real> read_all(const char* path) {
    std::vector<real> v;
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    real x;
    while (std::fread(&x, sizeof(real), 1, f) == 1) v.push_back(x);
    std::fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: compare_params A B TOL\n");
        return 2;
    }
    const std::vector<real> a = read_all(argv[1]), b = read_all(argv[2]);
    if (a.size() != b.size() || a.empty()) {
        std::printf("size mismatch: %zu vs %zu\n", a.size(), b.size());
        return 1;
    }
    double worst = 0;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::fmax(worst, std::fabs(double(a[i]) - double(b[i])));
    const double tol = std::atof(argv[3]);
    std::printf("max_abs_diff=%.3e (%zu params) %s\n", worst, a.size(), worst <= tol ? "PASS" : "FAIL");
    return worst <= tol ? 0 : 1;
}
