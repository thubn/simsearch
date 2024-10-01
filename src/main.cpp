#include "embedding_search.h"
#include <iostream>
#include <vector>
#include <random>

int main()
{
    EmbeddingSearch searcher;

    std::cout << "Hi World" << std::endl;

    // Load embeddings
    /*if (!searcher.load_safetensors("../data/embeddings.safetensors"))
    {
        return 1;
    }*/
    if (!searcher.load_json("../data/wiki_minilm.ndjson"))
    {
        return 1;
    }

    std::cout << "vector_size: " << searcher.getVectorSize() << "\nEmbeddings: " << searcher.getEmbeddings().size() << std::endl;
    std::cout << "first vector: " << searcher.getEmbeddings()[0][0] << std::endl;
    std::cout << "first vector: " << searcher.getEmbeddings()[0][1] << std::endl;
    std::cout << "second vector: " << searcher.getEmbeddings()[1][0] << std::endl;

    // Generate random index
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, searcher.getEmbeddings().size());
    auto random_index = distrib(gen);
    std::cout << "random index: " << random_index << std::endl;

    // Example query vector
    std::vector<float> query = searcher.getEmbeddings()[random_index]; // Adjust size to match your embeddings

    // Perform similarity search
    size_t k = 10; // Number of similar vectors to retrieve
    std::vector<std::pair<float, size_t>> results = searcher.similarity_search(query, k);

    auto sentences = searcher.getSentences();

    // Print results
    std::cout << "Top " << k << " similar vectors:" << std::endl;
    for (std::pair<float, size_t> idx : results)
    {
        std::cout << "Index: " << idx.second << "\tScore: " << idx.first << "\nSentence:\n"
                  << sentences[idx.second] << std::endl
                  << std::endl;
    }

    return 0;
}