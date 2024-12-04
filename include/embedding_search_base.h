#pragma once
#include "config_manager.h"
#include "embedding_io.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename VectorType, typename SimilarityType,
          typename RawType = float>
class EmbeddingSearchBase {
protected:
  std::vector<VectorType> embeddings;
  std::vector<std::string> sentences;
  // size_t vector_dim = 0;
  size_t num_vectors = 0;
  size_t vector_dim = 0;
  size_t padded_dim = 0;

public:
  virtual ~EmbeddingSearchBase() = default;

  // Common optimized methods
  virtual bool
  setEmbeddings(const std::vector<std::vector<float>> &input_vectors) = 0;

  // Common interface methods
  virtual std::vector<std::pair<SimilarityType, size_t>>
  similarity_search(const VectorType &query, size_t k) = 0;

  // Optional two-step search interface
  virtual std::vector<std::pair<SimilarityType, size_t>>
  similarity_search(const VectorType &query, size_t k,
                    std::vector<std::pair<int, size_t>> &searchIndexes) {
    throw std::runtime_error(
        "Two-step search not implemented for this searcher");
  }

  bool load(const std::string &filename, bool set_sentences = true,
            const int embedding_dim = 1024) {
    std::vector<std::vector<float>> temp_embeddings;
    std::vector<std::string> temp_sentences;
    bool result = false;
    auto start = std::chrono::high_resolution_clock::now();
    if (filename.ends_with(".safetensors")) {
      result = EmbeddingIO::load_safetensors(filename, temp_embeddings,
                                             temp_sentences);
    } else if (filename.ends_with(".ndjson")) {
      result =
          EmbeddingIO::load_json(filename, temp_embeddings, temp_sentences);
    } else if (filename.ends_with(".jsonl")) {
      result =
          EmbeddingIO::load_json2(filename, temp_embeddings, temp_sentences);
    } else if (filename.ends_with(".parquet")) {
      result = EmbeddingIO::load_parquet(filename, temp_embeddings,
                                         temp_sentences, embedding_dim);
    } else {
      throw std::runtime_error("Unsupported file format");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000;
    std::cout << "loading took " << time << "ms" << std::endl;
    num_vectors = temp_embeddings.size();
    this->setEmbeddings(temp_embeddings);
    if (set_sentences) {
      this->setSentences(temp_sentences);
    }
    return result;
  }

  // Common getters/setters that were duplicated across classes
  const std::vector<VectorType> &getEmbeddings() const { return embeddings; }
  const std::vector<std::string> &getSentences() const { return sentences; }
  bool setSentences(const std::vector<std::string> &s) {
    sentences.resize(s.size());
    sentences = s;
    return true;
  }
  size_t getVectorSize() const { return vector_dim; }
  size_t getNumVectors() const { return num_vectors; }
  size_t getVectorDim() const { return vector_dim; }
  size_t getPaddedDim() const { return padded_dim; }
  bool isInitialized() const { return num_vectors > 0; }

  // Common utility methods
  virtual void unsetEmbeddings() { embeddings.clear(); }
  virtual void unsetSentences() { sentences.clear(); }

protected:
  virtual SimilarityType cosine_similarity(const VectorType &a,
                                           const VectorType &b) = 0;
  // Common initialization logic
  bool initializeDimensions(const std::vector<std::vector<RawType>> &input) {
    if (input.empty() || input[0].empty()) {
      return false;
    }
    num_vectors = input.size();
    vector_dim = input[0].size();
    return true;
  }

  // Common validation methods
  virtual bool
  validateDimensions(const std::vector<std::vector<RawType>> &input,
                     std::string &error_message) = 0;
};

// Base class specifically for optimized implementations
template <typename VectorType, typename SimilarityType, typename StorageType>
class OptimizedEmbeddingSearchBase
    : public EmbeddingSearchBase<VectorType, SimilarityType> {
protected:
  const config::SearchConfig &config_;
  std::unique_ptr<StorageType[]> embedding_data;
  size_t vectors_per_embedding;

public:
  OptimizedEmbeddingSearchBase()
      : config_(ConfigRef::get()), vectors_per_embedding(0) {}

  // Common optimized methods
  // virtual bool setEmbeddings(const std::vector<std::vector<float>>
  // &input_vectors) = 0;

  // Common memory management methods
  StorageType *get_embedding_ptr(size_t index) {
    return embedding_data.get() + index * vectors_per_embedding;
  }

  const StorageType *get_embedding_ptr(size_t index) const {
    return embedding_data.get() + index * vectors_per_embedding;
  }

protected:
  SimilarityType cosine_similarity(const VectorType &a,
                                   const VectorType &b) override {
    throw std::runtime_error(
        "Default cosine function not implemented for optimized searchers. Use "
        "cosine_similarity_optimized instead");
  }

  virtual SimilarityType
  cosine_similarity_optimized(const StorageType *vec_a,
                              const StorageType *vec_b) const = 0;
  // Common allocation method
  bool allocateAlignedMemory(size_t total_size) {
    embedding_data.reset(static_cast<StorageType *>(std::aligned_alloc(
        config_.memory.alignmentSize, total_size * sizeof(StorageType))));
    return embedding_data != nullptr;
  }
};
