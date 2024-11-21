#include "embedding_search_float_int8.h"
#include "embedding_io.h"
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <omp.h>
#include <stdexcept>

bool EmbeddingSearchFloatInt8::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors) {
  num_vectors = input_vectors.size();
  vector_dim = input_vectors[0].size();

  embeddings.resize(num_vectors, std::vector<SimplifiedFloat>(vector_dim));

  for (int i = 0; i < num_vectors; i++) {
    for (int j = 0; j < vector_dim; j++) {
      embeddings[i][j] = SimplifiedFloat(input_vectors[i][j]);
    }
  }

  return true;
}

std::vector<std::pair<float, size_t>>
EmbeddingSearchFloatInt8::similarity_search(
    const std::vector<SimplifiedFloat> &query, size_t k) {
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

float EmbeddingSearchFloatInt8::cosine_similarity(
    const std::vector<SimplifiedFloat> &a,
    const std::vector<SimplifiedFloat> &b) {
  float dot_product = 0.0f;

  for (size_t i = 0; i < a.size(); ++i) {
    dot_product += (a[i] * b[i]).to_float();
  }

  return dot_product;
}