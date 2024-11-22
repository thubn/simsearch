#include "embedding_search_float16.h"
#include "embedding_io.h"
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <omp.h>
#include <stdexcept>

bool EmbeddingSearchFloat16::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors) {
  initializeDimensions(input_vectors);

  embeddings.resize(num_vectors, std::vector<std::bfloat16_t>(vector_dim));

  for (int i = 0; i < num_vectors; i++) {
    for (int j = 0; j < vector_dim; j++) {
      embeddings[i][j] = std::bfloat16_t(input_vectors[i][j]);
    }
  }

  return true;
}

std::vector<std::pair<float, size_t>> EmbeddingSearchFloat16::similarity_search(
    const std::vector<std::bfloat16_t> &query, size_t k) {
  if (query.size() != embeddings[0].size()) {
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<float, size_t>> similarities;
  similarities.reserve(embeddings.size());

  for (size_t i = 0; i < embeddings.size(); ++i) {
    float sim = cosine_similarity(query, embeddings[i]);
    similarities.emplace_back(sim, i);
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<float, size_t>>(similarities.begin(),
                                               similarities.begin() + k);
}

float EmbeddingSearchFloat16::cosine_similarity(
    const std::vector<std::bfloat16_t> &a,
    const std::vector<std::bfloat16_t> &b) {
  float dot_product = 0.0f;

  for (size_t i = 0; i < a.size(); ++i) {
    dot_product += a[i] * b[i];
  }

  return dot_product;
}