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
            PARQUET_ASSIGN_OR_THROW(
                infile,
                arrow::io::ReadableFile::Open(filename));

            // Create a ParquetFileReader instance
            std::unique_ptr<parquet::arrow::FileReader> reader;
            PARQUET_THROW_NOT_OK(
                parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

            // Read the entire file as a Table
            std::shared_ptr<arrow::Table> table;
            PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

            // Get column indices
            // const int title_idx = table->schema()->GetFieldIndex("title");
            const int text_idx = table->schema()->GetFieldIndex("formatted_text");

            // Get text column for sentences
            auto text_column = table->column(text_idx);

            // Initialize vectors
            const int64_t num_rows = table->num_rows();
            const int embedding_dim = 1024;
            embeddings.clear();
            sentences.clear();
            embeddings.reserve(num_rows);
            sentences.reserve(num_rows);

            // Process text column chunk by chunk
            for (int chunk_idx = 0; chunk_idx < text_column->num_chunks(); ++chunk_idx)
            {
                auto text_array = std::static_pointer_cast<arrow::StringArray>(text_column->chunk(chunk_idx));
                for (int64_t i = 0; i < text_array->length(); ++i)
                {
                    if (!text_array->IsNull(i))
                    {
                        sentences.push_back(text_array->GetString(i).substr(0, 1024));
                    }
                    else
                    {
                        sentences.push_back(""); // Handle null values
                    }
                }
            }

            // Process embeddings
            std::vector<float> current_embedding;
            current_embedding.reserve(embedding_dim);

            for (int64_t row = 0; row < num_rows; ++row)
            {
                current_embedding.clear();

                for (int j = 0; j < embedding_dim; ++j)
                {
                    std::string col_name = "embedding_" + std::to_string(j);
                    int col_idx = table->schema()->GetFieldIndex(col_name);
                    auto embedding_column = table->column(col_idx);
                    auto embedding_array = std::static_pointer_cast<arrow::FloatArray>(embedding_column->chunk(0));

                    if (!embedding_array->IsNull(row))
                    {
                        current_embedding.push_back(embedding_array->Value(row));
                    }
                    else
                    {
                        current_embedding.push_back(0.0f); // Handle null values
                    }
                }

                embeddings.push_back(current_embedding);
            }

            std::cout << "Loaded " << num_rows << " embeddings of dimension " << embedding_dim
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