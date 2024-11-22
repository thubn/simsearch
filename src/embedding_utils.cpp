#include "embedding_utils.h"
#include "aligned_types.h"
#include <cstdint>
#include <omp.h>
#include <stdexcept>

namespace EmbeddingUtils {
void convertSingleEmbeddingToAVX2(const std::vector<float> &input,
                                  avx2_vector &output, size_t vector_dim) {
  for (size_t j = 0; j < vector_dim; j++) {
    size_t k = j * 8; // Each AVX2 vector holds 8 floats
    // Since we're using aligned vectors, we can use aligned load
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
  for (size_t j = 0; j < vector_dim; j++) {
    size_t k = j * 32;

    // Create temporary array for int8 values
    std::vector<int8_t> temp_int8(32, 0);

    // Convert float values to int8
    for (int l = 0; l < 32; l++) {
      temp_int8[l] = static_cast<int8_t>(input[k + l] * 127);
    }

    // Load int8 values into AVX2 vector
    output[j] =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(temp_int8.data()));
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
} // namespace EmbeddingUtils