# Embedding Search Project

This project implements a simple vector similarity search for embeddings stored in a safetensors file.

## Building the Project

To build the project, simply run:

```
make
```

This will compile the source files and create an executable named `embedding_search` in the `bin` directory.

## Running the Program

Before running the program, make sure you have a safetensors file with your embeddings in the `data` directory.

To run the program:

```
./bin/embedding_search
```

The program will load the embeddings from `data/embeddings.safetensors`, perform a similarity search with a sample query vector, and print the indices of the top 5 most similar vectors.

## Cleaning the Build

To clean the build artifacts, run:

```
make clean
```

This will remove the `obj` and `bin` directories.