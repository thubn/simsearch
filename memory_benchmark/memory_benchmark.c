#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ARRAY_SIZE ((size_t)(1024ULL * 1024ULL * 256ULL * 1ULL)) // 12GB
#define NUM_ITERATIONS 16
#define ALIGN_SIZE 16 // For NEON alignment
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

double test_read_bandwidth_simd(float *array) {
  const unsigned long start_time = get_time_ns();

  // We'll keep these registers throughout the function
  register unsigned long offset1 asm("x9") = 0x20;
  register unsigned long offset2 asm("x10") = 0x40;
  register unsigned long offset3 asm("x11") = 0x60;
  register unsigned long offset4 asm("x12") = 0x80;
  register unsigned long offset5 asm("x13") = 0xa0;
  register unsigned long offset6 asm("x14") = 0xc0;
  register unsigned long offset7 asm("x15") = 0xe0;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    for (size_t i = 0; i < VEC_SIZE - 63; i += 64) {
      asm volatile(
          // Load base address and calculate all offsets
          "mov x1, %[addr]\n\t"
          "add x2, x1, x9\n\t"
          "add x3, x1, x10\n\t"
          "add x4, x1, x11\n\t"
          "add x5, x1, x12\n\t"
          "add x6, x1, x13\n\t"
          "add x7, x1, x14\n\t"
          "add x8, x1, x15\n\t"

          // Load all chunks
          "ld1 {v0.4s, v1.4s}, [x1]\n\t"
          "ld1 {v2.4s, v3.4s}, [x2]\n\t"
          "ld1 {v4.4s, v5.4s}, [x3]\n\t"
          "ld1 {v6.4s, v7.4s}, [x4]\n\t"
          "ld1 {v8.4s, v9.4s}, [x5]\n\t"
          "ld1 {v10.4s, v11.4s}, [x6]\n\t"
          "ld1 {v12.4s, v13.4s}, [x7]\n\t"
          "ld1 {v14.4s, v15.4s}, [x8]\n\t"
          :
          : [addr] "r"(&array[i])
          : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "v0", "v1", "v2",
            "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
            "v13", "v14", "v15", "memory");
    }
  }

  const unsigned long end_time = get_time_ns();
  return calc_bandwidth(start_time, end_time);
}

double test_read_bandwidth_simd_stride(float *array) {
  const unsigned long start_time = get_time_ns();

  // Keep these registers throughout the function
  register unsigned long offset1 asm("x9") = 0x1000;
  register unsigned long offset2 asm("x10") = 0x2000;
  register unsigned long offset3 asm("x11") = 0x3000;
  register unsigned long offset4 asm("x12") = 0x4000;
  register unsigned long offset5 asm("x13") = 0x5000;
  register unsigned long offset6 asm("x14") = 0x6000;
  register unsigned long offset7 asm("x15") = 0x7000;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    for (size_t i = 0; i < VEC_SIZE - TOTAL_STRIDE_DIST + 1;
         i += TOTAL_STRIDE_DIST) {
      for (size_t j = i; j < STRIDE_LENGTH + i; j += 8) {
        asm volatile("mov x1, %[addr]\n\t"
                     "add x2, x1, x9\n\t"
                     "add x3, x1, x10\n\t"
                     "add x4, x1, x11\n\t"
                     "add x5, x1, x12\n\t"
                     "add x6, x1, x13\n\t"
                     "add x7, x1, x14\n\t"
                     "add x8, x1, x15\n\t"

                     "ld1 {v0.4s, v1.4s}, [x1]\n\t"
                     "ld1 {v2.4s, v3.4s}, [x2]\n\t"
                     "ld1 {v4.4s, v5.4s}, [x3]\n\t"
                     "ld1 {v6.4s, v7.4s}, [x4]\n\t"
                     "ld1 {v8.4s, v9.4s}, [x5]\n\t"
                     "ld1 {v10.4s, v11.4s}, [x6]\n\t"
                     "ld1 {v12.4s, v13.4s}, [x7]\n\t"
                     "ld1 {v14.4s, v15.4s}, [x8]\n\t"
                     :
                     : [addr] "r"(&array[j])
                     : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "v0",
                       "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                       "v10", "v11", "v12", "v13", "v14", "v15", "memory");
      }
    }
  }

  const unsigned long end_time = get_time_ns();
  return calc_bandwidth(start_time, end_time);
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
