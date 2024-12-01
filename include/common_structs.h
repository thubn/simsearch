// common_structures.h
#pragma once

#include "aligned_types.h"
#include "embedding_search_avx2.h"
#include "embedding_search_binary.h"
#include "embedding_search_binary_avx2.h"
#include "embedding_search_float.h"
#include "embedding_search_float16.h"
#include "embedding_search_mapped_float.h"
#include "embedding_search_uint8_avx2.h"
#include "optimized_embedding_search_avx2.h"
#include "optimized_embedding_search_binary_avx2.h"
// #include "optimized_embedding_search_mapped_float.h"
#include "optimized_embedding_search_uint8_avx2.h"
#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace simsearch {

// Searcher types enum
enum SearcherType {
  F32,
  F32_PCA2,
  F32_PCA4,
  F32_PCA2x2,
  F32_PCA6,
  F32_PCA8,
  F32_PCA16,
  F32_PCA32,
  F32_AVX2,
  F32_AVX2_PCA8,
  F32_OAVX2,
  BINARY,
  BINARY_AVX2,
  BINARY_AVX2_PCA6,
  OBINAR_AVX2,
  OBAVX2_F32OAVX2,
  UINT8_AVX2,
  OUINT8_AVX2,
  FLOAT_INT8,
  FLOAT16,
  MAPPED_FLOAT,
  MAPPED_FLOAT2,
  NUM_SEARCHER_TYPES // Used to determine array sizes
};

// Struct to hold benchmark results
struct BenchmarkResults {
  std::vector<int64_t> times;
  std::vector<double> jaccardIndexes;
  std::vector<double> NDCG;

  BenchmarkResults()
      : times(NUM_SEARCHER_TYPES, -1), jaccardIndexes(NUM_SEARCHER_TYPES, 0),
        NDCG(NUM_SEARCHER_TYPES, 0) {}
};

// Structure to hold query information
struct Query {
  std::string query;            // Original query text
  std::string formatted_query;  // Preprocessed/formatted query text
  std::vector<float> embedding; // Query embedding vector

  Query() = default;
  Query(const std::string &q, const std::string &fq,
        const std::vector<float> &emb)
      : query(q), formatted_query(fq), embedding(emb) {}
};

// Structure for benchmark configuration
struct BenchmarkConfig {
  size_t k;                // Number of results to retrieve
  size_t runs;             // Number of benchmark runs
  size_t rescoring_factor; // Factor for two-step search rescoring

  BenchmarkConfig(size_t k = 25, size_t runs = 500, size_t rescoring = 50)
      : k(k), runs(runs), rescoring_factor(rescoring) {}
};

// Structure holding all searcher instances
class Searchers {
public:
  EmbeddingSearchFloat base;
  OptimizedEmbeddingSearchAVX2 pca2;
  OptimizedEmbeddingSearchAVX2 pca4;
  OptimizedEmbeddingSearchAVX2 pca8;
  OptimizedEmbeddingSearchAVX2 pca16;
  OptimizedEmbeddingSearchAVX2 pca32;
  EmbeddingSearchAVX2 avx2;
  EmbeddingSearchBinary binary;
  EmbeddingSearchBinaryAVX2 binary_avx2;
  EmbeddingSearchUint8AVX2 uint8_avx2;
  OptimizedEmbeddingSearchAVX2 oavx2;
  OptimizedEmbeddingSearchBinaryAVX2 obinary_avx2;
  OptimizedEmbeddingSearchUint8AVX2 ouint8_avx2;
  EmbeddingSearchFloat16 float16;
  EmbeddingSearchMappedFloat mappedFloat;
  // OptimizedEmbeddingSearchMappedFloat mappedFloat2;

  Searchers() = default;

  // Initialize base embeddings from file
  bool initBase(const std::string &filename, const int embedding_dim) {
    return base.load(filename, true, embedding_dim);
  }

  void initPca2() {
    std::cout << "Start loading of PCA2" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(2);
    pca2.setEmbeddings(temp.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of PCA2 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initPca4() {
    std::cout << "Start loading of PCA4" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(4);
    pca4.setEmbeddings(temp.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of PCA4 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initPca8() {
    std::cout << "Start loading of PCA8" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(8);
    pca8.setEmbeddings(temp.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of PCA8 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initPca16() {
    std::cout << "Start loading of PCA16" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(16);
    pca16.setEmbeddings(temp.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of PCA16 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initPca32() {
    std::cout << "Start loading of PCA32" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(32);
    pca32.setEmbeddings(temp.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of PCA32 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initAvx2() {
    std::cout << "Start loading of AVX2" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    avx2.setEmbeddings(base.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of AVX2 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initBinary() {
    std::cout << "Start loading of Binary" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    binary.setEmbeddings(base.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of Binary finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initBinary_avx2() {
    std::cout << "Start loading of BinaryAVX2" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    binary_avx2.setEmbeddings(base.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of BinaryAVX2 finished. Elapsed time: " << time
              << "ms" << std::endl;
  }
  void initUint8_avx2() {
    std::cout << "Start loading of Int8AVX2" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    uint8_avx2.setEmbeddings(base.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of Int8AVX2 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initOavx2() {
    std::cout << "Start loading of oAVX2" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    oavx2.setEmbeddings(base.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of oAVX2 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initObinary_avx2() {
    std::cout << "Start loading of oBinary" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    obinary_avx2.setEmbeddings(base.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of oBinary finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initOuint_avx2() {
    std::cout << "Start loading of oInt8" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    ouint8_avx2.setEmbeddings(base.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of oInt8 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initFloat16() {
    std::cout << "Start loading of Float16" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    float16.setEmbeddings(base.getEmbeddings());
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of Float16 finished. Elapsed time: " << time << "ms"
              << std::endl;
  }
  void initMappedFloat() {
    std::cout << "Start loading of mapped Float" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    mappedFloat.setEmbeddings(base.getEmbeddings(), 10.0);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    std::cout << "Loading of mapped Float finished. Elapsed time: " << time
              << "ms" << std::endl;
  }
  /*void initMappedFloat2() {
    mappedFloat2.setEmbeddings(base.getEmbeddings());
  }*/
};

void initializeSearchers(Searchers &searchers, const std::string &filename,
                         const int embedding_dim = 1024,
                         const bool init_pca = true,
                         const bool init_avx2 = true,
                         const bool init_binary = true,
                         const bool init_int8 = true,
                         const bool init_float16 = true,
                         const bool init_mf = true) {
  // Load base embeddings
  searchers.initBase(filename, embedding_dim);

  std::thread tOavx2;
  std::thread tObinary_avx2;
  std::thread tOuint_avx2;
  std::thread tFloat16;
  std::thread tMappedFloat;
  std::thread tPca2;
  std::thread tPca4;
  std::thread tPca8;
  std::thread tPca16;
  std::thread tPca32;

  // std::thread tAvx2(&Searchers::initAvx2, &searchers);
  // std::thread tBinary(&Searchers::initBinary, &searchers);
  // std::thread tBinary_avx2(&Searchers::initBinary_avx2, &searchers);
  // std::thread tUint8_avx2(&Searchers::initUint8_avx2, &searchers);
  if (init_avx2)
    tOavx2 = std::thread(&Searchers::initOavx2, &searchers);
  if (init_binary)
    tObinary_avx2 = std::thread(&Searchers::initObinary_avx2, &searchers);
  if (init_int8)
    tOuint_avx2 = std::thread(&Searchers::initOuint_avx2, &searchers);
  if (init_float16)
    tFloat16 = std::thread(&Searchers::initFloat16, &searchers);
  if (init_mf)
    tMappedFloat = std::thread(&Searchers::initMappedFloat, &searchers);
  // std::thread tMappedFloat2(&Searchers::initMappedFloat2, &searchers);
  if (init_pca) {
    tPca2 = std::thread(&Searchers::initPca2, &searchers);
    tPca4 = std::thread(&Searchers::initPca4, &searchers);
    tPca8 = std::thread(&Searchers::initPca8, &searchers);
    tPca16 = std::thread(&Searchers::initPca16, &searchers);
    tPca32 = std::thread(&Searchers::initPca32, &searchers);
  }

  // tAvx2.join();
  // tBinary.join();
  // tBinary_avx2.join();
  // tUint8_avx2.join();
  if (init_avx2)
    tOavx2.join();
  if (init_binary)
    tObinary_avx2.join();
  if (init_int8)
    tOuint_avx2.join();
  if (init_float16)
    tFloat16.join();
  if (init_mf)
    tMappedFloat.join();
  // tMappedFloat2.join();

  if (init_pca) {
    tPca2.join();
    tPca4.join();
    tPca8.join();
    tPca16.join();
    tPca32.join();
  }
}

} // namespace simsearch