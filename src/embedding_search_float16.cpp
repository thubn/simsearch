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

  embeddings.resize(num_vectors, std::vector<std::float16_t>(vector_dim));

  for (int i = 0; i < num_vectors; i++) {
    for (int j = 0; j < vector_dim; j++) {
      embeddings[i][j] = std::float16_t(input_vectors[i][j]);
    }
  }

  return true;
}

std::vector<std::pair<float, size_t>>
EmbeddingSearchFloat16::similarity_search(const std::vector<float> &query,
                                          size_t k) {
  std::vector<std::float16_t> float16_query(query.size());
  for (int i = 0; i < query.size(); i++) {
    float16_query[i] = std::float16_t(query[i]);
  }
  return similarity_search(float16_query, k);
}

std::vector<std::pair<float, size_t>> EmbeddingSearchFloat16::similarity_search(
    const std::vector<std::float16_t> &query, size_t k) {
  if (query.size() != embeddings[0].size()) {
    throw std::runtime_error("Query vector size does not match embedding size");
  }

  std::vector<std::pair<float, size_t>> similarities(embeddings.size());

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < embeddings.size(); ++i) {
    float sim = cosine_similarity(query, embeddings[i]);
    similarities[i] = std::make_pair(sim, i);
  }

  std::partial_sort(
      similarities.begin(), similarities.begin() + k, similarities.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  return std::vector<std::pair<float, size_t>>(similarities.begin(),
                                               similarities.begin() + k);
}

float EmbeddingSearchFloat16::cosine_similarity(
    const std::vector<std::float16_t> &a,
    const std::vector<std::float16_t> &b) {
  std::float16_t dot_product = std::float16_t(0.0f);

  for (size_t i = 0; i < a.size(); ++i) {
    dot_product += a[i] * b[i];
  }

  return dot_product;
}