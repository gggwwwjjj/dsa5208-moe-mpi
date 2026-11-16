#pragma once
// Storage for every array the compute backend touches (plan §5.5). CPU builds
// use ordinary heap memory; the CUDA backend allocates it with
// cudaMallocManaged (unified memory), so the same pointer works in host code
// (MoE routing, loss, MPI) and in kernels, and the driver migrates pages on
// demand. rvec = std::vector<real> in that memory.
#include <cstddef>
#include <new>
#include <vector>

#include "config.hpp"

namespace backend {
void* alloc_shared(std::size_t bytes);  // never returns nullptr (throws std::bad_alloc)
void free_shared(void* p) noexcept;
}  // namespace backend

template <class T>
struct SharedAlloc {
    using value_type = T;
    SharedAlloc() = default;
    template <class U>
    SharedAlloc(const SharedAlloc<U>&) {}
    T* allocate(std::size_t n) { return static_cast<T*>(backend::alloc_shared(n * sizeof(T))); }
    void deallocate(T* p, std::size_t) noexcept { backend::free_shared(p); }
};
template <class T, class U>
bool operator==(const SharedAlloc<T>&, const SharedAlloc<U>&) { return true; }
template <class T, class U>
bool operator!=(const SharedAlloc<T>&, const SharedAlloc<U>&) { return false; }

template <class T>
using svec = std::vector<T, SharedAlloc<T>>;
using rvec = svec<real>;
