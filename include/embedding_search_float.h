#pragma once
#include "embedding_search_base.h"
#include "embedding_io.h"

class EmbeddingSearchFloat
    : public EmbeddingSearchBase<std::vector<float>, float> {
public:
  EmbeddingSearchFloat() = default;

  bool load(const std::string &filename, bool set_sentences = true,
            const int embedding_dim = 1024) {
    bool result = false;
    auto start = std::chrono::high_resolution_clock::now();
    if (filename.ends_with(".safetensors")) {
      result = EmbeddingIO::load_safetensors(filename, embeddings, sentences);
    } else if (filename.ends_with(".ndjson")) {
      result = EmbeddingIO::load_json(filename, embeddings, sentences);
    } else if (filename.ends_with(".jsonl")) {
      result = EmbeddingIO::load_json2(filename, embeddings, sentences);
    } else if (filename.ends_with(".parquet")) {
      result = EmbeddingIO::load_parquet(filename, embeddings, sentences,
                                         set_sentences, embedding_dim);
    } else {
      throw std::runtime_error("Unsupported file format");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000;
    std::cout << "loading took " << time << "ms" << std::endl;
    initializeDimensions(embeddings);
    return result;
  }

  std::vector<std::pair<float, size_t>>
  similarity_search(const std::vector<float> &query, size_t k) override;

  bool pca_dimension_reduction(int factor);

  bool validateDimensions(const std::vector<std::vector<float>> &input,
                          std::string &error_message) override {
    if (input.empty()) {
      error_message = "Input vector is empty";
      return false;
    }
    return true;
  }

  bool setEmbeddings(
      const std::vector<std::vector<float>> &float_embeddings) override;

private:
  float cosine_similarity(const std::vector<float> &a,
                          const std::vector<float> &b);
};