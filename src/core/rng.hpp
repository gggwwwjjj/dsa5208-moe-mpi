#pragma once
// Stateless, portable random numbers. std::shuffle and std::uniform_*_distribution
// are implementation-defined (libc++ on macOS vs libstdc++ on Atlas), so they
// would give different batches on different machines. Everything random in the
// project (initial weights, epoch permutations, synthetic data) comes from here.
#include <cstdint>
#include <numeric>
#include <vector>

// splitmix64 finaliser
inline std::uint64_t mix64(std::uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Uniform in [0, 1) determined only by (seed, a, b).
inline double uniform01(std::uint64_t seed, std::uint64_t a, std::uint64_t b) {
    std::uint64_t h = mix64(seed ^ mix64(a ^ mix64(b)));
    return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);  // 53 bits
}

// Fisher-Yates permutation of 0..n-1 for one epoch; identical on every rank and platform.
inline std::vector<long> epoch_permutation(long n, std::uint64_t seed, long epoch) {
    std::vector<long> perm(static_cast<std::size_t>(n));
    std::iota(perm.begin(), perm.end(), 0L);
    for (long i = n - 1; i > 0; --i) {
        long j = static_cast<long>(uniform01(seed, 0x5eedULL + static_cast<std::uint64_t>(epoch),
                                             static_cast<std::uint64_t>(i)) * static_cast<double>(i + 1));
        std::swap(perm[static_cast<std::size_t>(i)], perm[static_cast<std::size_t>(j)]);
    }
    return perm;
}
