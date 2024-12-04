#include "embedding_utils.h"
#include "aligned_types.h"
#include <cstdint>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdexcept>

namespace EmbeddingUtils {
void convertSingleEmbeddingToAVX2(const std::vector<float> &input,
                                  avx2_vector &output, size_t vector_dim) {
  for (size_t j = 0; j < vector_dim; j++) {
    size_t k = j * 8; // Each AVX2 vector holds 8 floats
    // Unaligned load, input vector can be unaligned
    output[j] = _mm256_loadu_ps(&input[k]);
  }
}

void convertEmbeddingsToAVX2(const std::vector<std::vector<float>> &input,
                             std::vector<avx2_vector> &output,
                             size_t vector_dim) {
  size_t num_embeddings = input.size();
  output.resize(num_embeddings, avx2_vector(vector_dim));

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < num_embeddings; i++) {
    convertSingleEmbeddingToAVX2(input[i], output[i], vector_dim);
  }
}

bool validateAVX2Dimensions(const std::vector<std::vector<float>> &input,
                            std::string &error_message) {
  if (input.empty()) {
    error_message = "Input vector is empty";
    return false;
  }

  if (input[0].size() % 8 != 0) {
    error_message = "Input vector size must be a multiple of 8";
    return false;
  }

  return true;
}

size_t calculateBinaryAVX2VectorSize(size_t float_vector_size) {
  return ((float_vector_size + 63) / 64 + 3) / 4;
}

void convertSingleFloatToBinaryAVX2(const std::vector<float> &input,
                                    avx2i_vector &output, size_t vector_dim) {
  // Implementation for binary conversion
  for (size_t j = 0; j < vector_dim; j++) {
    uint64_t temp_bits[4] = {0, 0, 0, 0};
    size_t base_idx = j * 256;

    for (size_t chunk = 0; chunk < 4; chunk++) {
      for (size_t bit = 0; bit < 64; bit++) {
        size_t idx = base_idx + chunk * 64 + bit;
        if (idx < input.size() && input[idx] >= 0) {
          temp_bits[chunk] |= (1ULL << (63 - bit));
        }
      }
    }

    output[j] = _mm256_set_epi64x(temp_bits[3], temp_bits[2], temp_bits[1],
                                  temp_bits[0]);
  }
}

bool validateBinaryAVX2Dimensions(const std::vector<std::vector<float>> &input,
                                  std::string &error_message) {
  if (input.empty()) {
    error_message = "Input vector is empty";
    return false;
  }

  if (input[0].empty()) {
    error_message = "Input vectors cannot be empty";
    return false;
  }

  return true;
}

void convertSingleFloatToUint8AVX2(const std::vector<float> &input,
                                   avx2i_vector &output, size_t vector_dim) {
  for (size_t i = 0; i < vector_dim; i++) {
    std::vector<int8_t> temp(32, 0); // Initialize with zeros for padding

    // Only fill up to the actual dimension size
    for (size_t j = 0; j < 32 && (i * 32 + j) < input.size(); j++) {
      float val = input[i * 32 + j];
      temp[j] = static_cast<int8_t>(std::clamp(val * 127.0f, -127.0f, 127.0f));
    }

    output[i] =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(temp.data()));
  }
}

bool validateUint8AVX2Dimensions(const std::vector<std::vector<float>> &input,
                                 std::string &error_message) {
  if (input.empty()) {
    error_message = "Input vector is empty";
    return false;
  }

  if (input[0].empty()) {
    error_message = "Input vectors cannot be empty";
    return false;
  }

  if (input[0].size() % 32 != 0) {
    error_message = "Input vector size must be a multiple of 32";
    return false;
  }

  return true;
}

size_t calculateUint8AVX2VectorSize(size_t float_vector_size) {
  return float_vector_size / 32; // Each AVX2 vector holds 32 int8 values
}

// Function to sanitize UTF-8 text
std::string sanitize_utf8(const std::string &input) {
  std::string output;
  output.reserve(input.size());

  for (size_t i = 0; i < input.size();) {
    unsigned char c = input[i];

    if (c < 0x80) { // ASCII character
      if (c < 0x20 && c != '\n' && c != '\t' && c != '\r') {
        // Replace control characters except newline, tab, and carriage return
        output += ' ';
      } else {
        output += c;
      }
      i++;
      continue;
    }

    // Multi-byte UTF-8 sequence
    int length = 0;
    if ((c & 0xE0) == 0xC0)
      length = 2; // 2-byte sequence
    else if ((c & 0xF0) == 0xE0)
      length = 3; // 3-byte sequence
    else if ((c & 0xF8) == 0xF0)
      length = 4; // 4-byte sequence
    else {
      // Invalid UTF-8 sequence start
      output += ' ';
      i++;
      continue;
    }

    // Check if we have enough bytes for the sequence
    if (i + length > input.size()) {
      // Incomplete sequence
      output += ' ';
      i++;
      continue;
    }

    // Validate continuation bytes
    bool valid = true;
    for (int j = 1; j < length; j++) {
      if ((input[i + j] & 0xC0) != 0x80) {
        valid = false;
        break;
      }
    }

    if (valid) {
      // Copy valid UTF-8 sequence
      output.append(input.substr(i, length));
      i += length;
    } else {
      // Invalid sequence
      output += ' ';
      i++;
    }
  }

  return output;
}

void printMatrix(std::vector<std::vector<float>> m) {
  for (auto &row : m) {
    for (auto &cell : row) {
      std::cout << std::fixed << std::setprecision(3) << std::setw(5) << cell
                << " ";
    }
    std::cout << std::endl;
  }
}

bool pca_dimension_reduction(
    const int factor, const std::vector<std::vector<float>> &input_embeddings,
    std::vector<std::vector<float>> &result_embeddings,
    std::vector<std::vector<float>> &result_pca_matrix,
    std::vector<float> &result_mean) {
  // Convert data to Eigen matrix
  int rows = input_embeddings.size();
  int cols = input_embeddings[0].size();
  Eigen::MatrixXf matrix(rows, cols);
  int target_dim = cols / factor;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix(i, j) = input_embeddings[i][j];
    }
  }

  // Center the data
  Eigen::VectorXf mean = matrix.colwise().mean();
  matrix = matrix.rowwise() - mean.transpose();
  result_mean = std::vector<float>(mean.size());
  for (int i = 0; i < result_mean.size(); i++) {
    result_mean[i] = mean(i);
  }

  // Compute covariance matrix
  Eigen::MatrixXf cov = (matrix.transpose() * matrix) / (float)(rows - 1);

  // Compute eigendecomposition
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);

  // Sort eigenvectors by eigenvalues
  std::vector<std::pair<float, Eigen::VectorXf>> eigens;
  for (int i = 0; i < cols; ++i) {
    eigens.push_back({eig.eigenvalues()[i], eig.eigenvectors().col(i)});
  }

  std::sort(eigens.begin(), eigens.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  result_pca_matrix =
      std::vector<std::vector<float>>(cols, std::vector<float>(target_dim));

  // Select top k eigenvectors
  Eigen::MatrixXf pca_matrix(cols, target_dim);
  for (int i = 0; i < target_dim; ++i) {
    pca_matrix.col(i) = eigens[i].second;
    for (int j = 0; j < cols; ++j) {
      result_pca_matrix[j][i] = pca_matrix(j, i);
    }
  }
  // printMatrix(result_pca_matrix);

  // Project data onto principal components
  Eigen::MatrixXf reduced = matrix * pca_matrix;

  // Convert back to vector of vectors
  // std::vector<std::vector<float>> result(rows,
  // std::vector<float>(target_dim));
  result_embeddings =
      std::vector<std::vector<float>>(rows, std::vector<float>(target_dim));

  // #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < target_dim; ++j) {
      result_embeddings[i][j] = reduced(i, j);
    }
  }

  for (std::vector<float> &vec : result_embeddings) {
    float norm = EmbeddingUtils::calcNorm(vec);
    for (float &el : vec) {
      el = el / norm;
    }
  }

  return true;
}
std::vector<float> apply_pca_dimension_reduction_to_query(
    const std::vector<std::vector<float>> &pca_matrix,
    const std::vector<float> &mean, const std::vector<float> &query) {
  size_t org_vector_dim = pca_matrix.size();
  size_t vector_dim = pca_matrix[0].size();
  /*std::cout << "org_vector_dim: " << org_vector_dim
            << "\nvector_dim: " << vector_dim
            << "\nquery.size(): " << query.size() << std::endl;*/
  if (org_vector_dim != query.size()) {
    std::cout << "query_dim != org_vector_dim! org_vector_dim: "
              << org_vector_dim << " query_dim: " << query.size() << std::endl;
    exit(1);
  }
  Eigen::VectorXf eigen_query(org_vector_dim);
  Eigen::VectorXf eigen_mean(org_vector_dim);
  Eigen::MatrixXf eigen_pca_matrix(org_vector_dim, vector_dim);
  for (int i = 0; i < org_vector_dim; i++) {
    for (int j = 0; j < vector_dim; j++) {
      eigen_pca_matrix(i, j) = pca_matrix[i][j];
    }
    eigen_query(i) = query[i];
    eigen_mean(i) = mean[i];
  }

  Eigen::VectorXf centered_query = eigen_query - eigen_mean;
  Eigen::VectorXf eigen_result_query(vector_dim);
  eigen_result_query = eigen_pca_matrix.transpose() * centered_query;
  
  if (vector_dim != eigen_result_query.size()) {
    std::cout << "vector_dim != eigen_result_query.size(): " << vector_dim
              << " != " << eigen_result_query.size() << std::endl;
  }

  float norm = eigen_result_query.norm();
  eigen_result_query /= norm;

  std::vector<float> result_query(vector_dim);
  for (int i = 0; i < vector_dim; i++) {
    result_query[i] = eigen_result_query(i);
  }
  return result_query;
}
} // namespace EmbeddingUtils