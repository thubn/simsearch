#include "embedding_io.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/chunked_array.h>
#include <arrow/type_traits.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <omp.h>

using json = nlohmann::json;

namespace EmbeddingIO
{

    class BoundedQueue
    {
    private:
        std::queue<std::string> queue_;
        mutable std::mutex mutex_; // Mark mutex as mutable
        std::condition_variable not_full_;
        std::condition_variable not_empty_;
        size_t capacity_;
        bool finished_{false};

    public:
        explicit BoundedQueue(size_t capacity = 1000) : capacity_(capacity) {}

        void push(const std::string &value)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            not_full_.wait(lock, [this]
                           { return queue_.size() < capacity_ || finished_; });
            if (finished_)
                return;

            queue_.push(value);
            not_empty_.notify_one();
        }

        bool pop(std::string &value)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            not_empty_.wait(lock, [this]
                            { return !queue_.empty() || finished_; });

            if (queue_.empty() && finished_)
            {
                return false;
            }

            value = std::move(queue_.front());
            queue_.pop();
            not_full_.notify_one();
            return true;
        }

        void finish()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
            not_empty_.notify_all();
            not_full_.notify_all();
        }

        size_t size() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }
    };

    struct ThreadSafeEmbeddings
    {
        std::vector<std::vector<float>> embeddings;
        std::vector<std::string> sentences;
        std::mutex mutex;

        void add(std::vector<float> &&embedding, std::string &&sentence)
        {
            std::lock_guard<std::mutex> lock(mutex);
            embeddings.push_back(std::move(embedding));
            sentences.push_back(std::move(sentence));
        }
    };

    std::string readUTF8StringFromFile(const std::string &filename, size_t offset, uint64_t &length)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        file.seekg(offset);
        file.read(reinterpret_cast<char *>(&length), sizeof(length));

        std::string result(length, '\0');
        file.read(&result[0], length);

        if (!file)
        {
            throw std::runtime_error("Error reading from file: " + filename);
        }

        return result;
    }

    bool load_safetensors(const std::string &filename, std::vector<std::vector<float>> &embeddings, std::vector<std::string> &sentences)
    {
        try
        {
            uint64_t headerLength;
            auto headerJson = readUTF8StringFromFile(filename, 8, headerLength);

            json safetensorsHeader = json::parse(headerJson);

            uint64_t num_vectors = safetensorsHeader.at("shard_0").at("shape").at(0).get<int>();
            uint64_t vector_dim = safetensorsHeader.at("shard_0").at("shape").at(1).get<int>();
            uint64_t data_offset = safetensorsHeader.at("shard_0").at("data_offsets").at(0).get<int>();

            std::ifstream file(filename, std::ios::binary);
            if (!file)
            {
                throw std::runtime_error("Failed to open file: " + filename);
            }

            embeddings.resize(num_vectors, std::vector<float>(vector_dim));

            file.seekg(8 + headerLength + data_offset);

            for (auto &vec : embeddings)
            {
                file.read(reinterpret_cast<char *>(vec.data()), vector_dim * sizeof(float));
                if (!file)
                {
                    throw std::runtime_error("Error reading embedding data from file: " + filename);
                }
            }

            std::cout << "Loaded " << num_vectors << " embeddings of dimension " << vector_dim << " from " << filename << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in load_safetensors: " << e.what() << std::endl;
            return false;
        }
    }

    void process_lines_json(BoundedQueue &queue, ThreadSafeEmbeddings &result, std::atomic<int> &error_count)
    {
        std::string line;
        while (queue.pop(line))
        {
            try
            {
                json j = json::parse(line);
                result.add(
                    j.at("all-MiniLM-L6-v2").get<std::vector<float>>(),
                    j.at("body").get<std::string>());
            }
            catch (const std::exception &e)
            {
                error_count++;
                std::cerr << "Error processing line: " << e.what() << std::endl;
            }
        }
    }

    void process_lines_json2(BoundedQueue &queue, ThreadSafeEmbeddings &result, std::atomic<int> &error_count)
    {
        std::string line;
        while (queue.pop(line))
        {
            try
            {
                json j = json::parse(line);
                result.add(
                    j.at(1).at("data").at(0).at("embedding").get<std::vector<float>>(),
                    j.at(0).at("input").get<std::string>());
            }
            catch (const std::exception &e)
            {
                error_count++;
                std::cerr << "Error processing line: " << e.what() << std::endl;
            }
        }
    }

    bool load_json(const std::string &filename, std::vector<std::vector<float>> &embeddings, std::vector<std::string> &sentences)
    {
        try
        {
            std::ifstream file(filename);
            if (!file)
            {
                throw std::runtime_error("Failed to open file: " + filename);
            }

            // Initialize thread-safe structures with bounded queue
            constexpr size_t QUEUE_CAPACITY = 1000; // Adjust this based on your memory constraints
            BoundedQueue queue(QUEUE_CAPACITY);
            ThreadSafeEmbeddings result;
            std::atomic<int> error_count{0};

            // Create worker threads
            unsigned int num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> workers;
            workers.reserve(num_threads);

            // Start worker threads
            for (unsigned int i = 0; i < num_threads; ++i)
            {
                workers.emplace_back(process_lines_json, std::ref(queue), std::ref(result), std::ref(error_count));
            }

            // Read lines and add to queue (will block if queue is full)
            std::string line;
            int line_count = 0;
            while (std::getline(file, line))
            {
                queue.push(line);
                line_count++;

                if (line_count % 10000 == 0)
                {
                    std::cout << "Processed " << line_count << " lines, queue size: " << queue.size() << std::endl;
                }
            }

            // Signal completion to workers
            queue.finish();

            // Wait for all workers to finish
            for (auto &worker : workers)
            {
                worker.join();
            }

            // Move results to output parameters
            embeddings = std::move(result.embeddings);
            sentences = std::move(result.sentences);

            if (embeddings.empty())
            {
                throw std::runtime_error("No valid embeddings found in file: " + filename);
            }

            std::cout << "Loaded " << embeddings.size() << " embeddings of dimension "
                      << embeddings[0].size() << " from " << filename
                      << " (Errors: " << error_count << ")" << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in load_json: " << e.what() << std::endl;
            return false;
        }
    }

    bool load_json2(const std::string &filename, std::vector<std::vector<float>> &embeddings, std::vector<std::string> &sentences)
    {
        try
        {
            std::ifstream file(filename);
            if (!file)
            {
                throw std::runtime_error("Failed to open file: " + filename);
            }

            // Initialize thread-safe structures with bounded queue
            constexpr size_t QUEUE_CAPACITY = 1000; // Adjust this based on your memory constraints
            BoundedQueue queue(QUEUE_CAPACITY);
            ThreadSafeEmbeddings result;
            std::atomic<int> error_count{0};

            // Create worker threads
            unsigned int num_threads = std::thread::hardware_concurrency();
            std::vector<std::thread> workers;
            workers.reserve(num_threads);

            // Start worker threads
            for (unsigned int i = 0; i < num_threads; ++i)
            {
                workers.emplace_back(process_lines_json2, std::ref(queue), std::ref(result), std::ref(error_count));
            }

            // Read lines and add to queue (will block if queue is full)
            std::string line;
            int line_count = 0;
            while (std::getline(file, line))
            {
                queue.push(line);
                line_count++;

                if (line_count % 10000 == 0)
                {
                    std::cout << "Processed " << line_count << " lines, queue size: " << queue.size() << std::endl;
                }
            }

            // Signal completion to workers
            queue.finish();

            // Wait for all workers to finish
            for (auto &worker : workers)
            {
                worker.join();
            }

            // Move results to output parameters
            embeddings = std::move(result.embeddings);
            sentences = std::move(result.sentences);

            if (embeddings.empty())
            {
                throw std::runtime_error("No valid embeddings found in file: " + filename);
            }

            std::cout << "Loaded " << embeddings.size() << " embeddings of dimension "
                      << embeddings[0].size() << " from " << filename
                      << " (Errors: " << error_count << ")" << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in load_json2: " << e.what() << std::endl;
            return false;
        }
    }

    bool load_parquet(const std::string &filename, std::vector<std::vector<float>> &embeddings, std::vector<std::string> &sentences)
    {
        try
        {
            // Open the file
            std::shared_ptr<arrow::io::ReadableFile> infile;
            PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(filename));

            // Create a ParquetFileReader instance
            std::unique_ptr<parquet::arrow::FileReader> reader;
            PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

            // Get the file metadata
            std::shared_ptr<arrow::Schema> schema;
            PARQUET_THROW_NOT_OK(reader->GetSchema(&schema));

            auto file_metadata = reader->parquet_reader()->metadata();
            int64_t num_rows = file_metadata->num_rows();
            int num_row_groups = file_metadata->num_row_groups();
            const int embedding_dim = 1024;

            std::cout << "Total rows: " << num_rows << ", Row groups: " << num_row_groups << std::endl;

            // Create vectors for each thread to avoid contention
            int num_threads = omp_get_max_threads();
            std::vector<std::vector<std::vector<float>>> thread_embeddings(num_threads);
            std::vector<std::vector<std::string>> thread_sentences(num_threads);

            // Pre-calculate expected size for each thread
            size_t expected_size = num_rows / num_threads + 1;
            for (int i = 0; i < num_threads; i++)
            {
                thread_embeddings[i].reserve(expected_size);
                thread_sentences[i].reserve(expected_size);
            }

            // Create column selection vector
            std::vector<int> text_column{schema->GetFieldIndex("formatted_text")};
            std::vector<int> embedding_columns;
            for (int i = 0; i < embedding_dim; i++)
            {
                std::string col_name = "embedding_" + std::to_string(i);
                embedding_columns.push_back(schema->GetFieldIndex(col_name));
            }

// Parallel processing of row groups
#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic)
                for (int row_group = 0; row_group < num_row_groups; row_group++)
                {
                    // Get number of rows in this group
                    int64_t rows_in_group = file_metadata->RowGroup(row_group)->num_rows();

                    // Read text column for this row group
                    std::shared_ptr<arrow::Table> text_batch;
#pragma omp critical
                    {
                        PARQUET_THROW_NOT_OK(reader->ReadRowGroup(row_group, text_column, &text_batch));
                    }
                    auto text_array = std::static_pointer_cast<arrow::StringArray>(text_batch->column(0)->chunk(0));

                    // Read embedding columns for this row group
                    std::shared_ptr<arrow::Table> embedding_batch;
#pragma omp critical
                    {
                        PARQUET_THROW_NOT_OK(reader->ReadRowGroup(row_group, embedding_columns, &embedding_batch));
                    }

                    // Process each row in the group
                    for (int64_t i = 0; i < rows_in_group; i++)
                    {
                        // Get text
                        thread_sentences[thread_id].push_back(
                            text_array->IsNull(i) ? "" : text_array->GetString(i).substr(0, 128));

                        // Get embedding values
                        std::vector<float> current_embedding;
                        current_embedding.reserve(embedding_dim);

                        for (int j = 0; j < embedding_dim; j++)
                        {
                            auto embedding_array = std::static_pointer_cast<arrow::FloatArray>(
                                embedding_batch->column(j)->chunk(0));
                            current_embedding.push_back(
                                embedding_array->IsNull(i) ? 0.0f : embedding_array->Value(i));
                        }

                        thread_embeddings[thread_id].push_back(std::move(current_embedding));
                    }
                }
            }

            // Combine results from all threads
            size_t total_size = 0;
            for (const auto &thread_vec : thread_embeddings)
            {
                total_size += thread_vec.size();
            }

            embeddings.clear();
            sentences.clear();
            embeddings.reserve(total_size);
            sentences.reserve(total_size);

            for (int i = 0; i < num_threads; i++)
            {
                embeddings.insert(embeddings.end(),
                                  std::make_move_iterator(thread_embeddings[i].begin()),
                                  std::make_move_iterator(thread_embeddings[i].end()));
                sentences.insert(sentences.end(),
                                 std::make_move_iterator(thread_sentences[i].begin()),
                                 std::make_move_iterator(thread_sentences[i].end()));
            }

            std::cout << "Loaded " << embeddings.size() << " embeddings of dimension " << embedding_dim
                      << " from " << filename << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error in load_parquet: " << e.what() << std::endl;
            return false;
        }
    }

} // namespace EmbeddingIO