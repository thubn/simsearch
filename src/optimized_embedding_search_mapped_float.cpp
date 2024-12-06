#include "optimized_embedding_search_mapped_float.h"
#include "embedding_utils.h"
#include <algorithm>
#include <cstring>
#include <numeric>

struct PartitionInfo {
  float start;
  float end;
  float average;
};

// Calculate weight based on relative position and distribution type
double calculateWeight(double relative_pos, DistributionType dist_type,
                       double factor = 10.0) {
  switch (dist_type) {
  case DistributionType::UNIFORM:
    return 1.0;
  case DistributionType::LINEAR:
    return 1.0 - (std::abs(relative_pos) * 0.5);
  case DistributionType::QUADRATIC:
    return 1.0 - (relative_pos * relative_pos * 0.75);
  case DistributionType::CUBIC:
    return 1.0 - (relative_pos * relative_pos * relative_pos * 0.5);
  case DistributionType::GAUSSIAN:
    return std::exp(-relative_pos * relative_pos * factor);
  case DistributionType::COSINE:
    return std::cos(relative_pos * M_PI_2) * 0.5 + 0.5;
  default:
    return 1.0;
  }
}

std::vector<PartitionInfo>
partitionAndAverage(std::vector<float> &arr, int n_parts,
                    DistributionType dist_type = DistributionType::QUADRATIC,
                    double factor = 10.0) {
  try {
    if (arr.empty()) {
      throw std::invalid_argument("Input array is empty");
    }
    if (n_parts <= 0 || n_parts > arr.size()) {
      throw std::invalid_argument("Invalid number of parts");
    }

    // Sort the array
    std::sort(arr.begin(), arr.end());

    // Find indices for zero crossing
    auto zero_it = std::lower_bound(arr.begin(), arr.end(), 0.0f);
    size_t zero_index = std::distance(arr.begin(), zero_it);

    // Calculate number of negative and positive values
    size_t neg_count = zero_index;
    size_t pos_count = arr.size() - zero_index;

    // Determine partition split based on data distribution
    int neg_parts = static_cast<int>(
        (static_cast<double>(neg_count) / arr.size()) * n_parts);
    int pos_parts = n_parts - neg_parts;

    // Ensure at least one partition for each side if there's data
    if (neg_count > 0 && neg_parts == 0)
      neg_parts = 1;
    if (pos_count > 0 && pos_parts == 0)
      pos_parts = 1;
    pos_parts = n_parts - neg_parts;

    /*
    std::cout << "Distribution info:\n";
    std::cout << "Distribution type: " << static_cast<int>(dist_type) << "\n";
    std::cout << "Total elements: " << arr.size() << "\n";
    std::cout << "Negative elements: " << neg_count << " (" << neg_parts
              << " partitions)\n";
    std::cout << "Positive elements: " << pos_count << " (" << pos_parts
              << " partitions)\n\n";
    */

    // Calculate weights and sizes for both sides
    std::vector<size_t> partition_sizes(n_parts);

    // Handle negative partitions
    if (neg_parts > 0) {
      std::vector<double> weights(neg_parts);
      double total_weight = 0.0;

      // Calculate weights
      for (int i = 0; i < neg_parts; i++) {
        double relative_pos = (double(i) / double(neg_parts - 1)) - 1.0;
        // std::cout << "neg relative_pos: " << i << " " << relative_pos <<
        // std::endl;
        weights[i] = calculateWeight(relative_pos, dist_type, factor);
        total_weight += weights[i];
      }

      // Convert weights to sizes
      size_t elements_assigned = 0;
      for (int i = 0; i < neg_parts - 1; i++) {
        double proportion = weights[i] / total_weight;
        partition_sizes[i] =
            std::max(size_t(1), static_cast<size_t>(neg_count * proportion));
        elements_assigned += partition_sizes[i];
      }
      // Last negative partition gets remaining elements
      if (elements_assigned < neg_count) {
        partition_sizes[neg_parts - 1] = neg_count - elements_assigned;
      }
    }

    // Handle positive partitions
    if (pos_parts > 0) {
      std::vector<double> weights(pos_parts);
      double total_weight = 0.0;

      // Calculate weights
      for (int i = 0; i < pos_parts; i++) {
        double relative_pos = double(i) / double(pos_parts - 1);
        // std::cout << "pos relative_pos: " << i << " " << relative_pos <<
        // std::endl;
        weights[i] = calculateWeight(relative_pos, dist_type, factor);
        total_weight += weights[i];
      }

      // Convert weights to sizes
      size_t elements_assigned = 0;
      for (int i = 0; i < pos_parts - 1; i++) {
        double proportion = weights[i] / total_weight;
        partition_sizes[i + neg_parts] =
            std::max(size_t(1), static_cast<size_t>(pos_count * proportion));
        elements_assigned += partition_sizes[i + neg_parts];
      }
      // Last positive partition gets remaining elements
      if (elements_assigned < pos_count) {
        partition_sizes[n_parts - 1] = pos_count - elements_assigned;
      }
    }

    // Create and fill partitions
    std::vector<PartitionInfo> partitions(n_parts);
    size_t start_pos = 0;

    for (int i = 0; i < n_parts; ++i) {
      if (partition_sizes[i] == 0) {
        continue;
      }

      double sum = 0.0f;
      for (size_t j = 0; j < partition_sizes[i]; ++j) {
        sum += double(arr[start_pos + j]);
      }

      float average = partition_sizes[i] > 0 ? sum / partition_sizes[i] : 0.0f;

      partitions[i] = {arr[start_pos], arr[start_pos + partition_sizes[i] - 1],
                       average};

      /*
      // Print partition details
      std::cout << "Partition " << std::setw(2) << i << ": "
                << "size = " << std::setw(10) << partition_sizes[i]
                << " elements, range [" << std::fixed << std::setprecision(6)
                << partitions[i].start << ", " << partitions[i].end
                << "], avg = " << partitions[i].average << "\n";
      */

      start_pos += partition_sizes[i];
    }

    return partitions;
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
  std::cout << std::endl;
}

// Custom exception for out-of-range values
class ValueOutOfRangeException : public std::runtime_error {
public:
  ValueOutOfRangeException(float value, float min, float max)
      : std::runtime_error(
            "Value " + std::to_string(value) + " is outside the valid range [" +
            std::to_string(min) + ", " + std::to_string(max) + "]") {}
};

// Binary search function to find partition index for a value
uint8_t findPartitionIndex(const std::vector<PartitionInfo> &partitions,
                           float value) {
  if (partitions.empty()) {
    throw std::invalid_argument("Partition vector is empty");
  }

  // Check range
  if (value < partitions[0].start || value > partitions.back().end) {
    throw ValueOutOfRangeException(value, partitions[0].start,
                                   partitions.back().end);
  }

  int left = 0;
  int right = partitions.size() - 1;

  while (left <= right) {
    int mid = left + (right - left) / 2;

    // Check if value is in current partition
    if (value >= partitions[mid].start && value <= partitions[mid].end) {
      return static_cast<uint8_t>(mid);
    }

    // If value is less than partition start, search left half
    if (value < partitions[mid].start) {
      right = mid - 1;
    }
    // If value is greater than partition end, search right half
    else {
      left = mid + 1;
    }
  }

  // This should never happen if partitions are contiguous
  throw std::runtime_error(
      "Value falls between partitions - possible gap in partition definitions");
}

bool OptimizedEmbeddingSearchMappedFloat::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors) {
  return setEmbeddings(input_vectors, 10.0, DistributionType::GAUSSIAN);
}

bool OptimizedEmbeddingSearchMappedFloat::setEmbeddings(
    const std::vector<std::vector<float>> &input_vectors, double distrib_factor,
    DistributionType dist_type = DistributionType::GAUSSIAN) {
  std::string error_message;
  if (!validateDimensions(input_vectors, error_message)) {
    throw std::runtime_error(error_message);
  }

  if (!initializeDimensions(input_vectors)) {
    return false;
  }

  std::vector<float> flat_input = flattenMatrix(input_vectors);
  std::vector<PartitionInfo> partitions = partitionAndAverage(
      flat_input, 256, DistributionType::GAUSSIAN, distrib_factor);
  // printPartitions(partitions);

  std::vector<std::vector<uint8_t>> intermediate_embeddings;
  intermediate_embeddings.resize(num_vectors, std::vector<uint8_t>(vector_dim));

#pragma omp parallel for
  for (int i = 0; i < num_vectors; i++) {
    for (int j = 0; j < vector_dim; j++) {
      intermediate_embeddings[i][j] =
          findPartitionIndex(partitions, input_vectors[i][j]);
    }
  }
  for (int i = 0; i < partitions.size(); i++) {
    mapped_floats[i] = partitions[i].average;
  }

  // Calculate padding for int8 values (32 values per AVX2 vector)
  padded_dim = ((vector_dim + 31) / 32) * 32;
  vectors_per_embedding =
      padded_dim / 32; // Number of __m256i vectors needed per embedding

  // Allocate aligned memory for int8 indices
  size_t total_size = num_vectors * padded_dim;
  if (!allocateAlignedMemory(total_size)) {
    return false;
  }

  // Compute mapped float values
  // computeMappedFloats(input_vectors, distrib_factor, dist_type);

// Convert and store each vector as int8 indices
#pragma omp parallel for
  for (size_t i = 0; i < num_vectors; i++) {
    uint8_t *dest = get_embedding_ptr(i);
    dest = intermediate_embeddings[i].data();
    // convertToMappedFormat(input_vectors[i], dest);
  }

  return true;
}

void OptimizedEmbeddingSearchMappedFloat::convertToMappedFormat(
    const std::vector<float> &input, uint8_t *output) const {
  for (size_t i = 0; i < input.size(); i++) {
    // Binary search through partition boundaries to find appropriate index
    size_t idx =
        std::lower_bound(mapped_floats, mapped_floats + 256, input[i]) -
        mapped_floats;
    output[i] = static_cast<uint8_t>(std::min(idx, static_cast<size_t>(255)));
  }

  // Zero-pad remaining elements if necessary
  if (padded_dim > input.size()) {
    std::memset(output + input.size(), 0, padded_dim - input.size());
  }
}

std::vector<std::pair<float, size_t>>
OptimizedEmbeddingSearchMappedFloat::similarity_search(const avx2_vector &query,
                                                       size_t k) {
  throw std::runtime_error(
      "AVX2 vector input not supported in optimized version");
}

std::vector<std::pair<float, size_t>>
OptimizedEmbeddingSearchMappedFloat::similarity_search(
    const std::vector<float> &query, size_t k) {
  if (query.size() != vector_dim) {
    throw std::invalid_argument("Invalid query dimension");
  }

  // Convert query to int8 indices
  auto query_aligned = aligned_vector<uint8_t>(padded_dim);
  convertToMappedFormat(query, query_aligned.data());

  // Calculate similarities
  std::vector<std::pair<float, size_t>> results;
  results.reserve(num_vectors);

  for (size_t i = 0; i < num_vectors; i++) {
    float similarity =
        cosine_similarity_optimized(get_embedding_ptr(i), query_aligned.data());
    results.emplace_back(similarity, i);
  }

  // Partial sort to get top-k results
  if (results.size() > k) {
    std::partial_sort(
        results.begin(), results.begin() + k, results.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });
    results.resize(k);
  }

  return results;
}

float OptimizedEmbeddingSearchMappedFloat::cosine_similarity_optimized(
    const uint8_t *vec_a, const uint8_t *vec_b) const {
  __m256 sum = _mm256_setzero_ps();

  for (size_t i = 0; i < padded_dim; i += 32) {
    // Load 32 int8 indices
    __m256i indices_a =
        _mm256_load_si256(reinterpret_cast<const __m256i *>(vec_a + i));
    __m256i indices_b =
        _mm256_load_si256(reinterpret_cast<const __m256i *>(vec_b + i));

    // Process in chunks of 8 int8s
    for (int j = 0; j < 4; j++) {
      // Extract 8 indices from each vector
      __m128i chunk_a = _mm256_extracti128_si256(indices_a, j / 2);
      __m128i chunk_b = _mm256_extracti128_si256(indices_b, j / 2);
      if (j % 2) {
        chunk_a = _mm_unpackhi_epi8(chunk_a, chunk_a);
        chunk_b = _mm_unpackhi_epi8(chunk_b, chunk_b);
      } else {
        chunk_a = _mm_unpacklo_epi8(chunk_a, chunk_a);
        chunk_b = _mm_unpacklo_epi8(chunk_b, chunk_b);
      }

      // Convert 8 indices to 32-bit integers
      __m256i indices_a_32 = _mm256_cvtepi8_epi32(chunk_a);
      __m256i indices_b_32 = _mm256_cvtepi8_epi32(chunk_b);

      // Gather mapped float values using indices
      __m256 values_a = _mm256_i32gather_ps(mapped_floats, indices_a_32, 4);
      __m256 values_b = _mm256_i32gather_ps(mapped_floats, indices_b_32, 4);

      // Multiply and accumulate
      sum = _mm256_fmadd_ps(values_a, values_b, sum);
    }

    // Prefetch next cache lines
    _mm_prefetch(vec_a + i + 64, _MM_HINT_T0);
    _mm_prefetch(vec_b + i + 64, _MM_HINT_T0);
  }

  // Horizontal sum
  __m128 sum128 =
      _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);

  return _mm_cvtss_f32(sum128);
}

std::vector<float>
OptimizedEmbeddingSearchMappedFloat::getEmbedding(size_t index) const {
  if (index >= num_vectors) {
    throw std::out_of_range("Embedding index out of range");
  }

  std::vector<float> result(vector_dim);
  const uint8_t *embedding_indices = get_embedding_ptr(index);

  // Convert indices back to float values
  for (size_t i = 0; i < vector_dim; i++) {
    result[i] = mapped_floats[static_cast<uint8_t>(embedding_indices[i])];
  }

  return result;
}

bool OptimizedEmbeddingSearchMappedFloat::validateDimensions(
    const std::vector<std::vector<float>> &input, std::string &error_message) {
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