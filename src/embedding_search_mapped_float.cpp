#include "embedding_search_mapped_float.h"
#include "embedding_io.h"
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <omp.h>
#include <stdexcept>

struct PartitionInfo {
  float start;
  float end;
  float average;
};

std::vector<PartitionInfo> partitionAndAverage(std::vector<float> &arr,
                                               int n_parts) {
  try {
    // Pre-allocate vector to avoid reallocations
    std::vector<PartitionInfo> partitions;
    partitions.reserve(n_parts);

    // Input validation
    if (arr.empty()) {
      throw std::invalid_argument("Input array is empty");
    }
    if (n_parts <= 0 || n_parts > arr.size()) {
      throw std::invalid_argument("Invalid number of parts");
    }

    // Sort the array
    std::sort(arr.begin(), arr.end());

    // Calculate the size of each part
    int part_size = arr.size() / n_parts;
    int remainder = arr.size() % n_parts;

    // Current position in the array
    size_t current_pos = 0;

    // Process each part
    for (int i = 0; i < n_parts; ++i) {
      // Calculate size of current part (distribute remainder)
      int current_size = part_size + (i < remainder ? 1 : 0);

      // Calculate average for current part
      float sum = 0.0f;
      for (int j = 0; j < current_size; ++j) {
        sum += arr[current_pos + j];
      }
      float average = sum / current_size;

      // Store info
      partitions.push_back(
          {arr[current_pos], arr[current_pos + current_size - 1], average});

      // Update position
      current_pos += current_size;
    }

    return partitions;
  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed in partitionAndAverage: " << e.what()
              << '\n';
    std::cerr << "Available system memory might be insufficient\n";
    throw; // Re-throw the exception
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    throw;
  }
}

std::vector<float> flattenMatrix(const std::vector<std::vector<float>> &input) {
  std::vector<float> flattened;
  // size_t total_size = input.size() * input[0].size();
  try {
    size_t total_size = 0;
    for (const auto &row : input) {
      total_size += row.size();
    }
    flattened.reserve(total_size);
    for (const auto &row : input) {
      flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed in flattenMatrix: " << e.what()
              << '\n';
    std::cerr << "Available system memory might be insufficient\n";
    throw; // Re-throw the exception
  }
}

std::string floatToHex(float value) {
  union {
    float f;
    uint32_t i;
  } u;
  u.f = value;
  std::stringstream ss;
  ss << "0x" << std::hex << std::uppercase << u.i;
  return ss.str();
}

void printPartitions(const std::vector<PartitionInfo> &partitions) {
  // Print human-readable results
  std::cout << std::fixed
            << std::setprecision(6); // Increased from 2 to 6 decimal places
  std::cout << "Partition information:\n";
  for (int i = 0; i < partitions.size(); ++i) {
    std::cout << "Part " << (i + 1) << ": [" << std::setw(12)
              << partitions[i].start << ", " // Added setw for alignment
              << std::setw(12) << partitions[i].end << "] "
              << "Average: " << std::setw(12) << partitions[i].average
              << " (Hex: " << floatToHex(partitions[i].average) << ")\n";
  }
  std::cout << "\n";

  // Print C++ array initialization with hex values
  std::cout << "C++ array initialization with hex values:\n\n";

  // Print starts array
  std::cout << "float starts[] = {";
  for (size_t i = 0; i < partitions.size(); ++i) {
    std::cout << std::setw(12) << partitions[i].start << "f /*"
              << floatToHex(partitions[i].start) << "*/";
    if (i < partitions.size() - 1)
      std::cout << ", ";
  }
  std::cout << "};\n";

  // Print ends array
  std::cout << "float ends[] = {";
  for (size_t i = 0; i < partitions.size(); ++i) {
    std::cout << std::setw(12) << partitions[i].end << "f /*"
              << floatToHex(partitions[i].end) << "*/";
    if (i < partitions.size() - 1)
      std::cout << ", ";
  }
  std::cout << "};\n";

  // Print averages array
  std::cout << "float averages[] = {";
  for (size_t i = 0; i < partitions.size(); ++i) {
    std::cout << std::setw(12) << partitions[i].average << "f /*"
              << floatToHex(partitions[i].average) << "*/";
    if (i < partitions.size() - 1)
      std::cout << ", ";
  }
  std::cout << "};\n";
}

bool EmbeddingSearchMappedFloat::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors) {
  num_vectors = input_vectors.size();
  vector_dim = input_vectors[0].size();

  try {
    std::cout << "setembeddings1" << std::endl;
    std::vector<float> flat_input = flattenMatrix(input_vectors);
    std::cout << "setembeddings2" << std::endl;
    std::vector<PartitionInfo> partitions =
        partitionAndAverage(flat_input, 256);
    std::cout << "setembeddings3" << std::endl;
    printPartitions(partitions);
    std::cout << "setembeddings4" << std::endl;
  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed in setEmbeddings: " << e.what()
              << '\n';
    std::cerr << "Available system memory might be insufficient\n";
    throw; // Re-throw the exception
  }
  exit(0);

  return true;
}

std::vector<std::pair<float, size_t>>
EmbeddingSearchMappedFloat::similarity_search(const std::vector<uint8_t> &query,
                                              size_t k) {
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

float EmbeddingSearchMappedFloat::cosine_similarity(
    const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
  float dot_product = 0.0f;

  for (size_t i = 0; i < a.size(); ++i) {
    dot_product += a[i] * b[i];
  }

  return dot_product;
}