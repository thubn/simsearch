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
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(2);
    pca2.setEmbeddings(temp.getEmbeddings());
  }
  void initPca4() {
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(4);
    pca4.setEmbeddings(temp.getEmbeddings());
  }
  void initPca8() {
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(8);
    pca8.setEmbeddings(temp.getEmbeddings());
  }
  void initPca16() {
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(16);
    pca16.setEmbeddings(temp.getEmbeddings());
  }
  void initPca32() {
    EmbeddingSearchFloat temp;
    temp.setEmbeddings(base.getEmbeddings());
    temp.pca_dimension_reduction(32);
    pca32.setEmbeddings(temp.getEmbeddings());
  }
  void initAvx2() { avx2.setEmbeddings(base.getEmbeddings()); }
  void initBinary() { binary.setEmbeddings(base.getEmbeddings()); }
  void initBinary_avx2() { binary_avx2.setEmbeddings(base.getEmbeddings()); }
  void initUint8_avx2() { uint8_avx2.setEmbeddings(base.getEmbeddings()); }
  void initOavx2() { oavx2.setEmbeddings(base.getEmbeddings()); }
  void initObinary_avx2() { obinary_avx2.setEmbeddings(base.getEmbeddings()); }
  void initOuint_avx2() { ouint8_avx2.setEmbeddings(base.getEmbeddings()); }
  void initFloat16() { float16.setEmbeddings(base.getEmbeddings()); }
  void initMappedFloat() {
    mappedFloat.setEmbeddings(base.getEmbeddings(), 10.0);
  }
  /*void initMappedFloat2() {
    mappedFloat2.setEmbeddings(base.getEmbeddings());
  }*/
};

void initializeSearchers(Searchers &searchers, const std::string &filename,
                         const bool initPca = false, const int embedding_dim = 1024) {
  // Load base embeddings
  searchers.initBase(filename, embedding_dim);

  // std::thread tAvx2(&Searchers::initAvx2, &searchers);
  // std::thread tBinary(&Searchers::initBinary, &searchers);
  // std::thread tBinary_avx2(&Searchers::initBinary_avx2, &searchers);
  // std::thread tUint8_avx2(&Searchers::initUint8_avx2, &searchers);
  std::thread tOavx2(&Searchers::initOavx2, &searchers);
  std::thread tObinary_avx2(&Searchers::initObinary_avx2, &searchers);
  std::thread tOuint_avx2(&Searchers::initOuint_avx2, &searchers);
  // std::thread tFloat16(&Searchers::initFloat16, &searchers);
  std::thread tMappedFloat(&Searchers::initMappedFloat, &searchers);
  // std::thread tMappedFloat2(&Searchers::initMappedFloat2, &searchers);
  if (initPca) {
    std::thread tPca2(&Searchers::initPca2, &searchers);
    std::thread tPca4(&Searchers::initPca4, &searchers);
    std::thread tPca8(&Searchers::initPca8, &searchers);
    std::thread tPca16(&Searchers::initPca16, &searchers);
    std::thread tPca32(&Searchers::initPca32, &searchers);
    tPca2.join();
    tPca4.join();
    tPca8.join();
    tPca16.join();
    tPca32.join();
  }

  // tAvx2.join();
  // tBinary.join();
  // tBinary_avx2.join();
  // tUint8_avx2.join();
  tOavx2.join();
  tObinary_avx2.join();
  tOuint_avx2.join();
  // tFloat16.join();
  tMappedFloat.join();
  // tMappedFloat2.join();
}

} // namespace simsearch