// aligned_types.h
#pragma once
#include <arm_neon.h>
#include <memory>
#include <vector>

namespace AlignedTypes {
// Custom allocator for aligned memory
template <typename T> class AlignedAllocator {
public:
  using value_type = T;
  static constexpr size_t alignment = 16; // AVX2 needs 32-byte alignment

  AlignedAllocator() noexcept {}
  template <class U> AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

  T *allocate(size_t n) {
    if (n == 0)
      return nullptr;
    void *ptr = std::aligned_alloc(alignment, n * sizeof(T));
    if (!ptr)
      throw std::bad_alloc();
    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, size_t) noexcept { std::free(p); }
};

template <typename T, typename U>
bool operator==(const AlignedAllocator<T> &,
                const AlignedAllocator<U> &) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(const AlignedAllocator<T> &,
                const AlignedAllocator<U> &) noexcept {
  return false;
}

// Aligned vector type
template <typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

// Convenience type aliases for AVX2 vectors
using avx2_vector = aligned_vector<float32x4_t>;
using avx2i_vector = aligned_vector<int32x4_t>;
} // namespace AlignedTypes

// Bring types into global namespace
using avx2_vector = AlignedTypes::avx2_vector;
using avx2i_vector = AlignedTypes::avx2i_vector;
template <typename T> using aligned_vector = AlignedTypes::aligned_vector<T>;