#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ARRAY_SIZE ((size_t)(1024ULL * 1024ULL * 1024ULL * 12ULL)) // 12GB
#define NUM_ITERATIONS 16
#define ALIGN_SIZE 32 // For AVX2 alignment
#define NUM_STRIDES 8
#define STRIDE_LENGTH 1024 * 1
#define VEC_SIZE ARRAY_SIZE / sizeof(float)
#define TOTAL_STRIDE_DIST NUM_STRIDES *STRIDE_LENGTH

// Function to measure time in nanoseconds
__attribute__((noinline)) unsigned long get_time_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (unsigned long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

__attribute__((noinline)) double calc_bandwidth(const unsigned long start_time,
                                                const unsigned long end_time) {
  double seconds = (end_time - start_time) / 1e9;
  double bytes_processed = (double)ARRAY_SIZE * NUM_ITERATIONS;
  return (bytes_processed / (1024 * 1024 * 1024)) / seconds; // GB/s
}

// SIMD read bandwidth test using AVX2
double test_read_bandwidth_simd_stride(float *array) {

  const unsigned long start_time = get_time_ns();

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    for (size_t i = 0; i < VEC_SIZE - TOTAL_STRIDE_DIST + 1;
         i += TOTAL_STRIDE_DIST) {
      // prevent loop unrolling
#pragma GCC unroll 1
      for (size_t j = i; j < STRIDE_LENGTH + i; j += 8) {
        asm volatile( //
            "vmovaps (%0), %%ymm0\n\t"
            "vmovaps 0x1000(%0), %%ymm1\n\t" // offset 1 * 4096 bytes
            "vmovaps 0x2000(%0), %%ymm2\n\t" // offset 2 * 4096 bytes
            "vmovaps 0x3000(%0), %%ymm3\n\t" // offset 3 * 4096 bytes
            "vmovaps 0x4000(%0), %%ymm4\n\t" // offset 4 * 4096 bytes
            "vmovaps 0x5000(%0), %%ymm5\n\t" // offset 5 * 4096 bytes
            "vmovaps 0x6000(%0), %%ymm6\n\t" // offset 6 * 4096 bytes
            "vmovaps 0x7000(%0), %%ymm7"     // offset 7 * 4096 bytes
            :
            : "r"(&array[j])                  //
            : "ymm0", "ymm1", "ymm2", "ymm3", //
              "ymm4", "ymm5", "ymm6", "ymm7");
      }
    }
  }

  unsigned long end_time = get_time_ns();
  return calc_bandwidth(start_time, end_time); // GB/s
}

double test_read_bandwidth_simd(float *array) {

  const unsigned long start_time = get_time_ns();

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    // prevent loop unrolling
#pragma GCC unroll 1
    for (size_t i = 0; i < VEC_SIZE - 63; i += 64) {
      asm volatile(                      //
          "vmovaps (%0), %%ymm0\n\t"     // offset 1 avx2 vec
          "vmovaps 0x20(%0), %%ymm1\n\t" // offset 32 bytes
          "vmovaps 0x40(%0), %%ymm2\n\t" // offset 64 bytes
          "vmovaps 0x60(%0), %%ymm3\n\t" // ...
          "vmovaps 0x80(%0), %%ymm4\n\t"
          "vmovaps 0xa0(%0), %%ymm5\n\t"
          "vmovaps 0xc0(%0), %%ymm6\n\t"
          "vmovaps 0xe0(%0), %%ymm7"
          :
          : "r"(&array[i])                  //
          : "ymm0", "ymm1", "ymm2", "ymm3", //
            "ymm4", "ymm5", "ymm6", "ymm7");
    }
  }

  const unsigned long end_time = get_time_ns();

  return calc_bandwidth(start_time, end_time); // GB/s
}

int main() {
  // Allocate aligned memory for SIMD operations
  float *simd_array;

  if (posix_memalign((void **)&simd_array, ALIGN_SIZE, ARRAY_SIZE) != 0) {
    printf("Memory allocation failed!\n");
    return 1;
  }
  printf("%lu GB of Memory allocated. Initializing...\n",
         ARRAY_SIZE / (1024 * 1024 * 1024));

  // Initialize arrays
  memset(simd_array, 0, ARRAY_SIZE);
  printf("Memory initialized.\n");
  printf("Memory Bandwidth Tests:\n");

  // SIMD bandwidth tests
  printf("SIMD (AVX2) Tests:\n");
  double simd_read = test_read_bandwidth_simd(simd_array);
  printf("SIMD Read Bandwidth                        : %.2f GB/s\n", simd_read);
  simd_read = test_read_bandwidth_simd_stride(simd_array);
  printf("SIMD Read Bandwidth with stride length %lu: %.2f GB/s\n",
         STRIDE_LENGTH * sizeof(float), simd_read);

  free(simd_array);

  return 0;
}
