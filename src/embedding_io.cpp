#include "embedding_io.h"
#include <arrow/array.h>
#include <arrow/chunked_array.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <queue>
#include <stdexcept>
#include <thread>

using json = nlohmann::json;

namespace EmbeddingIO {

class BoundedQueue {
private:
  std::queue<std::string> queue_;
  mutable std::mutex mutex_; // Mark mutex as mutable
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  size_t capacity_;
  bool finished_{false};

public:
  explicit BoundedQueue(size_t capacity = 1000) : capacity_(capacity) {}

  void push(const std::string &value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock,
                   [this] { return queue_.size() < capacity_ || finished_; });
    if (finished_)
      return;

    queue_.push(value);
    not_empty_.notify_one();
  }

  bool pop(std::string &value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return !queue_.empty() || finished_; });

    if (queue_.empty() && finished_) {
      return false;
    }

    value = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
  }

  void finish() {
    std::lock_guard<std::mutex> lock(mutex_);
    finished_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }
};

struct ThreadSafeEmbeddings {
  std::vector<std::vector<float>> embeddings;
  std::vector<std::string> sentences;
  std::mutex mutex;

  void add(std::vector<float> &&embedding, std::string &&sentence) {
    std::lock_guard<std::mutex> lock(mutex);
    embeddings.push_back(std::move(embedding));
    sentences.push_back(std::move(sentence));
  }
};

std::string readUTF8StringFromFile(const std::string &filename, size_t offset,
                                   uint64_t &length) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  file.seekg(offset);
  file.read(reinterpret_cast<char *>(&length), sizeof(length));

  std::string result(length, '\0');
  file.read(&result[0], length);

  if (!file) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return result;
}

bool load_safetensors(const std::string &filename,
                      std::vector<std::vector<float>> &embeddings,
                      std::vector<std::string> &sentences) {
  try {
    uint64_t headerLength;
    auto headerJson = readUTF8StringFromFile(filename, 8, headerLength);

    json safetensorsHeader = json::parse(headerJson);

    uint64_t num_vectors =
        safetensorsHeader.at("shard_0").at("shape").at(0).get<int>();
    uint64_t vector_dim =
        safetensorsHeader.at("shard_0").at("shape").at(1).get<int>();
    uint64_t data_offset =
        safetensorsHeader.at("shard_0").at("data_offsets").at(0).get<int>();

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    embeddings.resize(num_vectors, std::vector<float>(vector_dim));

    file.seekg(8 + headerLength + data_offset);

    for (auto &vec : embeddings) {
      file.read(reinterpret_cast<char *>(vec.data()),
                vector_dim * sizeof(float));
      if (!file) {
        throw std::runtime_error("Error reading embedding data from file: " +
                                 filename);
      }
    }

    std::cout << "Loaded " << num_vectors << " embeddings of dimension "
              << vector_dim << " from " << filename << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error in load_safetensors: " << e.what() << std::endl;
    return false;
  }
}

void process_lines_json(BoundedQueue &queue, ThreadSafeEmbeddings &result,
                        std::atomic<int> &error_count) {
  std::string line;
  while (queue.pop(line)) {
    try {
      json j = json::parse(line);
      result.add(j.at("all-MiniLM-L6-v2").get<std::vector<float>>(),
                 j.at("body").get<std::string>());
    } catch (const std::exception &e) {
      error_count++;
      std::cerr << "Error processing line: " << e.what() << std::endl;
    }
  }
}

void process_lines_json2(BoundedQueue &queue, ThreadSafeEmbeddings &result,
                         std::atomic<int> &error_count) {
  std::string line;
  while (queue.pop(line)) {
    try {
      json j = json::parse(line);
      result.add(
          j.at(1).at("data").at(0).at("embedding").get<std::vector<float>>(),
          j.at(0).at("input").get<std::string>());
    } catch (const std::exception &e) {
      error_count++;
      std::cerr << "Error processing line: " << e.what() << std::endl;
    }
  }
}

bool load_json(const std::string &filename,
               std::vector<std::vector<float>> &embeddings,
               std::vector<std::string> &sentences) {
  try {
    std::ifstream file(filename);
    if (!file) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    // Initialize thread-safe structures with bounded queue
    constexpr size_t QUEUE_CAPACITY =
        1000; // Adjust this based on your memory constraints
    BoundedQueue queue(QUEUE_CAPACITY);
    ThreadSafeEmbeddings result;
    std::atomic<int> error_count{0};

    // Create worker threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    // Start worker threads
    for (unsigned int i = 0; i < num_threads; ++i) {
      workers.emplace_back(process_lines_json, std::ref(queue),
                           std::ref(result), std::ref(error_count));
    }

    // Read lines and add to queue (will block if queue is full)
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
      queue.push(line);
      line_count++;

      if (line_count % 10000 == 0) {
        std::cout << "Processed " << line_count
                  << " lines, queue size: " << queue.size() << std::endl;
      }
    }

    // Signal completion to workers
    queue.finish();

    // Wait for all workers to finish
    for (auto &worker : workers) {
      worker.join();
    }

    // Move results to output parameters
    embeddings = std::move(result.embeddings);
    sentences = std::move(result.sentences);

    if (embeddings.empty()) {
      throw std::runtime_error("No valid embeddings found in file: " +
                               filename);
    }

    std::cout << "Loaded " << embeddings.size() << " embeddings of dimension "
              << embeddings[0].size() << " from " << filename
              << " (Errors: " << error_count << ")" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error in load_json: " << e.what() << std::endl;
    return false;
  }
}

bool load_json2(const std::string &filename,
                std::vector<std::vector<float>> &embeddings,
                std::vector<std::string> &sentences) {
  try {
    std::ifstream file(filename);
    if (!file) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    // Initialize thread-safe structures with bounded queue
    constexpr size_t QUEUE_CAPACITY =
        1000; // Adjust this based on your memory constraints
    BoundedQueue queue(QUEUE_CAPACITY);
    ThreadSafeEmbeddings result;
    std::atomic<int> error_count{0};

    // Create worker threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    // Start worker threads
    for (unsigned int i = 0; i < num_threads; ++i) {
      workers.emplace_back(process_lines_json2, std::ref(queue),
                           std::ref(result), std::ref(error_count));
    }

    // Read lines and add to queue (will block if queue is full)
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
      queue.push(line);
      line_count++;

      if (line_count % 10000 == 0) {
        std::cout << "Processed " << line_count
                  << " lines, queue size: " << queue.size() << std::endl;
      }
    }

    // Signal completion to workers
    queue.finish();

    // Wait for all workers to finish
    for (auto &worker : workers) {
      worker.join();
    }

    // Move results to output parameters
    embeddings = std::move(result.embeddings);
    sentences = std::move(result.sentences);

    if (embeddings.empty()) {
      throw std::runtime_error("No valid embeddings found in file: " +
                               filename);
    }

    std::cout << "Loaded " << embeddings.size() << " embeddings of dimension "
              << embeddings[0].size() << " from " << filename
              << " (Errors: " << error_count << ")" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error in load_json2: " << e.what() << std::endl;
    return false;
  }
}

bool load_parquet(const std::string &filename,
                  std::vector<std::vector<float>> &embeddings,
                  std::vector<std::string> &sentences,
                  int /* num_threads unused */) {
  try {
    // Configuration and initialization
    arrow::MemoryPool *pool = arrow::default_memory_pool();
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(filename));

    auto reader_properties = parquet::ReaderProperties(pool);
    reader_properties.enable_buffered_stream();
    reader_properties.set_buffer_size(4 * 1024 * 1024); // 4MB buffer

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &reader));

    // Get metadata and setup
    std::shared_ptr<arrow::Schema> schema;
    PARQUET_THROW_NOT_OK(reader->GetSchema(&schema));
    auto file_metadata = reader->parquet_reader()->metadata();
    int64_t num_rows = file_metadata->num_rows();
    int num_row_groups = file_metadata->num_row_groups();
    const int embedding_dim = 1024;

    embeddings.clear();
    sentences.clear();
    embeddings.reserve(num_rows);
    sentences.reserve(num_rows);

    // Prepare column indices
    std::vector<int> text_column{schema->GetFieldIndex("formatted_text")};
    std::vector<int> embedding_columns;
    embedding_columns.reserve(embedding_dim);
    for (int i = 0; i < embedding_dim; i++) {
      embedding_columns.push_back(
          schema->GetFieldIndex("embedding_" + std::to_string(i)));
    }

    // Structure to hold batch data
    struct BatchData {
      std::shared_ptr<arrow::Table> text_batch;
      std::shared_ptr<arrow::Table> embedding_batch;
      int64_t rows_in_group;
    };

    // Async reading function
    auto read_row_group = [&](int row_group) -> BatchData {
      BatchData batch;
      batch.rows_in_group = file_metadata->RowGroup(row_group)->num_rows();

      PARQUET_THROW_NOT_OK(
          reader->ReadRowGroup(row_group, text_column, &batch.text_batch));
      PARQUET_THROW_NOT_OK(reader->ReadRowGroup(row_group, embedding_columns,
                                                &batch.embedding_batch));

      return batch;
    };

    // Process function
    auto process_batch = [&](const BatchData &batch) {
      auto text_array = std::static_pointer_cast<arrow::StringArray>(
          batch.text_batch->column(0)->chunk(0));

      std::vector<float> current_embedding;
      current_embedding.reserve(embedding_dim);
      std::string text_buffer;
      text_buffer.reserve(256);

      for (int64_t i = 0; i < batch.rows_in_group; i++) {
        // Process text
        if (text_array->IsNull(i)) {
          sentences.emplace_back();
        } else {
          text_buffer = text_array->GetString(i);
          if (text_buffer.length() > 128) {
            text_buffer.resize(128);
          }
          std::replace(text_buffer.begin(), text_buffer.end(), '\n', ' ');
          sentences.emplace_back(std::move(text_buffer));
        }

        // Process embedding
        current_embedding.clear();
        for (int j = 0; j < embedding_dim; j++) {
          auto embedding_array = std::static_pointer_cast<arrow::FloatArray>(
              batch.embedding_batch->column(j)->chunk(0));
          current_embedding.push_back(
              embedding_array->IsNull(i) ? 0.0f : embedding_array->Value(i));
        }
        embeddings.emplace_back(std::move(current_embedding));
      }
    };

    // Main async processing loop
    std::future<BatchData> next_batch;

    // Start first async read
    if (num_row_groups > 0) {
      next_batch = std::async(std::launch::async, read_row_group, 0);
    }

    // Process row groups with async I/O
    for (int row_group = 0; row_group < num_row_groups; row_group++) {
      // Get current batch
      BatchData current_batch = next_batch.get();

      // Start next async read if there are more row groups
      if (row_group + 1 < num_row_groups) {
        next_batch =
            std::async(std::launch::async, read_row_group, row_group + 1);
      }

      // Process current batch while next batch is being read
      process_batch(current_batch);

      if (row_group % 100 == 0) {
        std::cout << "Processed row group " << row_group << " of "
                  << num_row_groups << std::endl;
      }
    }

    std::cout << "Loaded " << embeddings.size() << " embeddings of dimension "
              << embedding_dim << " from " << filename << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "Error in load_parquet: " << e.what() << std::endl;
    return false;
  }
}

} // namespace EmbeddingIO