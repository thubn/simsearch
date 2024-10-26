// embedding_search_float.cpp
#include "embedding_search_float.h"
#include "embedding_io.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <omp.h>

bool EmbeddingSearchFloat::load(const std::string &filename)
{
    // Determine file type and call appropriate loader
    if (filename.ends_with(".safetensors"))
    {
        return EmbeddingIO::load_safetensors(filename, embeddings, sentences);
    }
    else if (filename.ends_with(".ndjson"))
    {
        return EmbeddingIO::load_json(filename, embeddings, sentences);
    }
    else if (filename.ends_with(".jsonl"))
    {
        return EmbeddingIO::load_json2(filename, embeddings, sentences);
    }
    else
    {
        throw std::runtime_error("Unsupported file format");
    }
}

bool EmbeddingSearchFloat::pca_dimension_reduction(int target_dim)
{
    // Convert data to Eigen matrix
    int rows = embeddings.size();
    int cols = embeddings[0].size();
    Eigen::MatrixXf matrix(rows, cols);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix(i, j) = embeddings[i][j];
        }
    }

    // Center the data
    Eigen::VectorXf mean = matrix.colwise().mean();
    matrix = matrix.rowwise() - mean.transpose();

    // Compute covariance matrix
    Eigen::MatrixXf cov = (matrix.transpose() * matrix) / (float)(rows - 1);

    // Compute eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);

    // Sort eigenvectors by descending eigenvalues
    std::vector<std::pair<float, Eigen::VectorXf>> eigens;
    for (int i = 0; i < cols; ++i)
    {
        eigens.push_back({eig.eigenvalues()[i], eig.eigenvectors().col(i)});
    }
    std::sort(eigens.begin(), eigens.end(),
              [](const auto &a, const auto &b)
              { return a.first > b.first; });

    // Select top k eigenvectors
    Eigen::MatrixXf pca_matrix(cols, target_dim);
    for (int i = 0; i < target_dim; ++i)
    {
        pca_matrix.col(i) = eigens[i].second;
    }

    // Project data onto principal components
    Eigen::MatrixXf reduced = matrix * pca_matrix;

    // Convert back to vector of vectors
    std::vector<std::vector<float>> result(rows, std::vector<float>(target_dim));
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < target_dim; ++j)
        {
            result[i][j] = reduced(i, j);
        }
    }

    embeddings.resize(result.size(), std::vector<float>(result[0].size()));
    embeddings = result;
    vector_size = target_dim;
    return true;
}

std::vector<std::pair<float, size_t>> EmbeddingSearchFloat::similarity_search(const std::vector<float> &query, size_t k)
{
    if (query.size() != embeddings[0].size())
    {
        throw std::runtime_error("Query vector size does not match embedding size");
    }

    std::vector<std::pair<float, size_t>> similarities;
    similarities.reserve(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); ++i)
    {
        float sim = cosine_similarity(query, embeddings[i]);
        similarities.emplace_back(sim, i);
    }

    std::partial_sort(similarities.begin(), similarities.begin() + k, similarities.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

    return std::vector<std::pair<float, size_t>>(similarities.begin(), similarities.begin() + k);
}

float EmbeddingSearchFloat::cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
{
    float dot_product = 0.0f;
    float mag_a = 0.0f;
    float mag_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i)
    {
        dot_product += a[i] * b[i];
        mag_a += a[i] * a[i];
        mag_b += b[i] * b[i];
    }

    return dot_product / (std::sqrt(mag_a) * std::sqrt(mag_b));
}