
# Introduction

This chapter creates and overview of Vector Similarity Search, its
importance in the data driven world, and outlines the motivation for
this research.

## Motivation

### The Power of Vector Similarity Search

Everyday massive amounts of digital data like texts, documents, images,
videos, etc. are created. Tomorrow even more. A large part of that data
is unstructured. This presents a challenge: How can we search
efficiently through this growing amount of information? The traditional
way would be using keyword-based search, where you search the data for
matching words or strings. But method does only works on text data and
doesn't work on a semantic level. \[put example here\]

This is where vector similarity search comes in. It is able to compare
the semantic meaning of a text based search queries to a database of
images, texts, etc. This enables you, for example, to search your local
photo gallery for pictures containing cats with the simple query 'cat'.

Vector Similarity Search works by converting the input data (texts,
images, \...) into embeddings, which are just high dimensional -
numerical vectors. Encoded in these vectors are for example the meaning
of its encoded text or the visual details of a picture. Recommendation
systems of streaming platforms, social networks and ecommerce platforms
are also often using vector similarity search.

### The need for optimization

Considering our world is becoming increasingly mobile first a new
problem arises. While Vector Similarity Search is currently mostly run
in datacenters, it is very expensive on resources like processing power
and memory. But there are valid reasons for running vector similarity
search on resource constrained mobile devices. Like a user searching
their photo gallery without having to upload all their data to a cloud
provider to offload computation onto the datacenters.

This creates the fundamental question: How can we perform vector
similarity search on devices with limited resources while maintaining as
much speed and accuracy as possible?

## Research Objectives

This thesis addresses the previously mentioned optimization challenges
for resource constrained devices by investigating and developing
optimization techniques for Vector Similarity Search. The following
approaches will be examined and fine-tuned:

-   Quantization methods which can decrease memory usage and increase
    speed at the cost of accuracy

-   Vector dimension reduction also can decrease memory usage and
    increase speed at the cost of accuracy

-   Two-Step approach to balance speed and accuracy

-   SIMD optimization to accelerate calculations by using modern
    hardware features

## Structure

\[TODO\]

# Theoretical Background {#chapter:kap2}

This chapter provides more background information about embeddings and
their properties. The mathematical theory of vector similarity is also
explained. This is important to fully understand the later chapters.

## Vector Similarity Search

### Vector embeddings and their properties

As mentioned in the introduction, vector embeddings are the numerical
representations of the inherent properties of the encoded data in an
n-dimensional vector space. The data is encoded using an embedding
model. Models are specific for a datatype. There are models for
text-embeddings, picture-embeddings, etc. The resulting vector dimension
depends on the model itself. The most advanced text models use up to
8192 dimensions. [@muennighoff2023mtebmassivetextembedding] Different
models produce also different embedding data on the same input data. The
common number format used to handle these embeddings is float32. It
guarantees high accuracy, probably way more than needed, at the cost of
high processing power and high memory usage. Because we do not want to
only compare two vectors but search for the vector most similar to
vector $a$ in a set of vectors. That means, that the computer has to
load an array of vectors into memory. One float uses 4bytes and if one
embedding has 1024 dimensions it will use `4bytes * 1024 = 4 KB` of
memory.

::: example
Text-to-Vector Mapping Consider how semantically similar phrases are
mapped to similar vectors in the embedding space: $$\begin{tabular}{rcl}
            \texttt{"Hello world"} & $\mapsto$ & $[0.1, -0.3, 0.8, \ldots, 0.4]$ \\[0.5em]
            \texttt{"Hi universe"} & $\mapsto$ & $[0.2, -0.2, 0.7, \ldots, 0.5]$ \\
        \end{tabular}$$ Notice how these greetings, which have similar
semantic meaning, are mapped to nearby points in the vector space, as
illustrated in Figure [2.1](#fig:vector-embeddings){reference-type="ref"
reference="fig:vector-embeddings"}.
:::

<figure id="fig:vector-embeddings">

<figcaption>Visualization of similar phrases mapped to nearby vectors
(projected onto 2D for illustration)</figcaption>
</figure>

::: remark
Dimensionality Note We typically work with high-dimensional vectors
(e.g., 768 or 1024 dimensions), the 2D visualization above just
illustrates how semantically similar texts are closer together in the
vector space.
:::

### Cosine similarity and other similarity metrics

Vector similarity is defined as the angle between two vectors. Perfect
similarity means the angle between the vectors is $0\degree$. No
similarity at all means the vectors point exactly in the opposite
direction with an angle of $180\degree$ (or $\pi$).

#### Dot product

Given two vectors $a$ and $b$ with $n$ dimensions the dot product is
calculated as follows:
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i = a_1b_1 + a_2b_2 + \cdots + a_nb_n$$
The dot product gives the following information of the relation of two
vectors:

-   positive dot product $\rightarrow$ angle is $<90\degree$

-   dot product close to zero $\rightarrow$ angle is $=90\degree$

-   negative dot product $\rightarrow$ angle is $>90\degree$

The dot product already helps to roughly estimate the similarity of two
vectors.

#### Cosine similarity {#sec:cosinetodot}

The cosine similarity is derived from the dot product:
$$\mathbf{a} \cdot \mathbf{b} = \abs{a} \abs{b} \cos(\theta)$$ Which
means the norms of both vectors are multiplied with each other and then
multiplied with the cosine of the angle between the vectors. This
equitation can be rearranged in the following way to get the cosine
similarity function:
$$\cos(\theta) = \frac{a \cdot b}{\abs{a} \abs{b}} = \frac{\sum_{i=1}^n a_i b_i}{\sqrt{\sum_{i=1}^n a_i^2} \cdot \sqrt{\sum_{i=1}^n b_i^2}}$$
It's easy to see that the output values range from $-1$ to $1$. The
Vectors in the cosine function are automatically normalized and one can
not only roughly estimate the angle between the vectors, like with the
dot product, but tell the exact angle. It's also possible to compare
different angles of different vector pairs and tell which pair is the
most similar with the cosine similarity.

-   $\cos(\pi + 2n \pi) = -1$$\rightarrow$ vectors point in exactly
    opposite directions ($180\degree$)

-   $\cos(\frac{1}{2} \pi + n \pi) = 0$$\rightarrow$ vectors are
    perpendicular ($90\degree$)

-   $\cos(2n\pi) = 1$$\rightarrow$ vectors point in the same directions
    ($0\degree$)

#### more about real world scenarios/examples?

## Hardware considerations

### Resource constraints on mobile devices

\[TODO\]

-   Typical server: 256GB+ RAM, 64+ cores

-   High-end phone: 8GB RAM, 8 cores

-   Mid-range phone: 4GB RAM, 6 cores

### SIMD instructions and AVX2

Single instruction, multiple data, abbreviated by SIMD, is a type of
processor organization that operates using "a master instruction applied
over a vector of related operands." [@simd_flynn] This type of processor
organization excells especially vector/array processing with minimal
branching operations, large data parallelism and limited communication
needs. Considering vector similarity search involves operations on large
vectors using SIMD seems fitting. In comparison using multiprocessor
systems (MIMD, multiple instruction, multiple data) is less optimal,
because of the increased communication overhead between processors. On
top of that many modern processors (mobile arm, desktop -, mobile - and
server x86-64 processors) support SIMD and enables them to process the
data more efficiently in terms of power usage and computation time.
 [@comparingsimdonx8664andarm64; @simdtoaccelerateperformance; @khadem2023vectorprocessingmobiledevicesbenchmark]

Advanced Vector Extensions (AVX) are SIMD instructions for the x86-64
architecture. They operate on 256bit registers and enables the processor
to do math operations on them. They can for example multiply 8 32bit
floats with one instruction. AVX2 adds even more instructions. AVX2
works on most desktop and laptop processors released after 2015. AVX2
will be used to optimize the vector similarity search program.

### Memory hierarchy and access patterns

In most cases of vector similarity search one search query gets compared
to every embedding to get the best matching embeddings for this query.
This implies that embedding vectors are loaded sequentually from memory.
When we consider the memory hierarchy of modern systems is
$$\texttt{non-volatile memory > main memory > L3 > L2 > L1 cache > CPU registers}$$
with size and access time of the memory decreasing drastically from left
to right as seen in [2.1](#tab:memory-latencies){reference-type="ref"
reference="tab:memory-latencies"}. The search query most likely stays in
cache all the time, because its always one of the two vectors that gets
accessed during search. The predictable access patterns of the embedding
vectors enables either automatic preloading of the soon to be used
vectors by the CPU prefetcher or loading them manually by using a
prefetch instruction, while computing similarity of the current
embedding.

But CPU cache can vary depending on the processor architecture and an
effective prefetching strategy for one cpu might be slower for another.
Cache is always relatively small so cache pollution by prefetching to
much data has to be avoided.

::: {#tab:memory-latencies}
  Memory Level     Access Time (CPU Cycles)         Access Time (ns)
  -------------- ---------------------------- ----------------------------
  L1 Cache         $\sim$`<!-- -->`{=html}4     $\sim$`<!-- -->`{=html}1
  L2 Cache        $\sim$`<!-- -->`{=html}12     $\sim$`<!-- -->`{=html}3
  L3 Cache        $\sim$`<!-- -->`{=html}80    $\sim$`<!-- -->`{=html}20
  RAM             $\sim$`<!-- -->`{=html}400   $\sim$`<!-- -->`{=html}100

  : Memory hierarchy access latencies. [@memoryaccesslatency]
  Understanding these timing differences explains why memory access
  patterns significantly impact performance.
:::

## Evaluation Metrics

### Jaccard Index

The Jaccard index is a statistical metric suitible for gauging the
similarity of two sets. It is defined as the size of the intersection
divided by the size of the union of the two sets:
$$J(A,B) = \frac{\abs{A \cap B}}{\abs{A \cup B}}$$ The definition
implies that $0 \leq J(A,B) \leq 1$. The Jaccard index of two sets that
match exactly is $1$ and $0$ for two sets that have no matching
elements. The Jaccard Index will be one metric used for comparing the
accuracy of different optimization strategies by comparing the result
set to a baseline result that used no optimization techniques.

### Normaized Discounted Cumulative Gain (NDCG)

The Discounted Cumulative Gain (DCG) is an evaluation metric for
information retrieval systems. It assumes that documents found later in
the results i.e. results that are less similar than the first results,
are less valuable to the users. As a result it uses a logarithmic
discount factor to reduce the weight of documents appearing lower in the
results. Mathematical definition [@ndcg]:
$$DCG(p) = \sum_{i=1}^{p} \frac{rel_i}{log_2(i+1)}$$ $rel_i$ is the
relevance of the element with index $i$. $log_2(i+1)$ is the logarithmic
discount factor. The relevance can be either assigned manually by a
human judging the relavance of the result or by calculating it by
comparing it to an optimal result set. The algorithm used here calcultes
the relavance score based on the position difference. Perfect match gets
a score of 1. There is an exponential decay of the lost relevance for
large differences. It uses $log_2(i+2)$ instead of $log_2(i+1)$ because
in the algorithm the index $i$ starts at $0$.

::: algorithm
Excerpt from algorithm used to calculate the NDCG.ndcg_algo

``` {.c++ language="C++"}
// Calculate DCG
double dcg = 0.0;
// k is the size of the result sets
for (size_t i = 0; i < k; ++i) {
    auto it = truthPositions.find(prediction[i].second);
    if (it != truthPositions.end()) {
    // Calculate relevance score based on position difference
    double posDiff = abs(static_cast<double>(it->second) - i);
    double relevance = exp(-posDiff / k); // exp decay
    dcg += relevance / log2(i + 2); // DCG formula
    }
}
```
:::

The normalized DCG (NDCG) is calculated by dividing the the DCG score by
the ideal DCG possible for that query:
$$IDCG = \sum_{i=1}^{p} \frac{1}{log_2(i+1)}$$ -
$$NDCG = \frac{DCG}{IDCG}$$ The NDCG produces scores between $0$ and
$1$, where $1$ represents the ideal result. This property allows the
comparison between different queries. In conclusion the NDCG is a
relevant metric for judging the performance of the different
optimization techniques that will analyzed in this thesis. It accounts
for both document relevance and rank position. The relevance is not just
binary 0 and 1 like it is for the jaccard index. It also models
realistic user behaviour by discounting documents appearing further back
in the results.

# Implementation

## Software Architecture

<figure id="fig:reldiag_simsearch">

<figcaption>Simplified class relation diagram of the simsearch c++
program. p-virt mean the function is a pure virtual
function</figcaption>
</figure>

That Program that implements and benchmarks the different vector
similarity search optimizations is written in modern C++. A simplified
diagram of the class hierarchy can be seen
in [3.1](#fig:reldiag_simsearch){reference-type="ref"
reference="fig:reldiag_simsearch"}.

### Design philosophy

Overall the system is realized through template based class hierarchies.
The base functionality is implemented in the base class like the loading
of embeddings from disk. Functions like `cosine_similarity` are
separated into the specialized classes because that's these functions
are the one that will be optimized. Furthermore, the classes are
implemented in a type-safe manner. This way type errors when using
different vector formats (float, binary, int8) are caught at compile
time rather than runtime. Also, many benchmark parameters are
configurable when running the program from command line.

### Class hierarchy

The mentioned class hierarchy allows implementing the different vector
similarity search implementations without having duplicates of code.
This also guarantees that the classes have a consistent interface. There
are two abstract main classes which provide the foundation for all
implementations: *EmbeddingSearchBase* and
*OptimizedEmbeddingSearchBase*. The *EmbeddingIO* class allows loading
of embeddings from the disk. Different formats are supported, but most
of these are non-standard. The *load* function in the
*EmbeddingSearchBase* class uses this class. After it has loaded the
embeddings from the disk it passes the float32 vectors to the classes
corresponding *setEmbeddings* implementation. This function converts the
vectors into the expected format. For the classes that inherit from the
*EmbeddingSearchBase* class the embeddings are stored in a vector of
*VectorType*. Where *VectorType* can be a vector of floats, vector of
integers or an aligned vector datatype. One element of *VectorType*
represents one embedding and the vector of these are interpreted as the
list or array of embeddings.

There also is the *PyEmbeddingSearch* class. This class is used to
create python bindings. These bindings allow to use the different
searcher implementations in python. There one can write their own
queries and search for similar embeddings with the C++ implementation.

#### Memory management strategies {#memorymanagement}

The *AlignedTypes* class defines the memory aligned vector types
*avx2_vector* and *avx2i_vector*. They represent a memory aligned vector
of the *\_\_m256* and *\_\_m256i* datatype
respectively [\[alg:vectortypesAVX2\]](#alg:vectortypesAVX2){reference-type="ref"
reference="alg:vectortypesAVX2"}. The main purpose of this class is to
ensure proper memory alignment (done by
[\[alg:AlignedAllocator\]](#alg:AlignedAllocator){reference-type="ref"
reference="alg:AlignedAllocator"}) for the SIMD operations. It is used
by the *EmbeddingSearchAVX2* and *EmbeddingSearchUint8AVX2* classes.
This allows these classes to store the AVX2 vectors in a memory aligned
manner, which in turn enables these classes to use aligned loads into
AVX registers which is faster than the unaligned load AVX instruction.

::: algorithm
*AlignedAllocator* class templateAlignedAllocator

``` {.c++ language="C++"}
template <typename T>
class AlignedAllocator {
    static constexpr size_t alignment = 32; // AVX2 needs 32B alignment
    T* allocate(size_t n) {
        void* ptr = std::aligned_alloc(alignment, n * sizeof(T));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* p, size_t) noexcept {
        std::free(p);
    }
};
```
:::

::: algorithm
Specialized vector types for AVX2vectortypesAVX2

``` {.c++ language="C++"}
template <typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;
// Vector of 256-bit float vectors
using avx2_vector = aligned_vector<__m256>;
// Vector of 256-bit integer vectors
using avx2i_vector = aligned_vector<__m256i>;
```
:::

The optimized classes also use aligned memory to store the vectors. But
in these classes a single contiguous block of memory is allocated to
store the embeddings. Which allows better usage of the CPU cache. Memory
is managed
manually [\[alg:embeddingdata\]](#alg:embeddingdata){reference-type="ref"
reference="alg:embeddingdata"}. It allows direct pointer arithmetic for
even faster access. This approach removes the minimal overhead that
*std::vector* has. This also enables the possibility to use memory
prefetching, because the memory address of vectors accessed in the
future is more predictable.

::: algorithm
Excerpt from *OptimizedEmbeddingSearchBase*embeddingdata

``` {.c++ language="C++"}
std::unique_ptr<StorageType[]> embedding_data;
// Used in derived classes:
StorageType* get_embedding_ptr(size_t index) {
    return embedding_data.get() + index * vectors_per_embedding;
}
bool allocateAlignedMemory(size_t total_size) {
    embedding_data.reset(static_cast<StorageType *>(std::aligned_alloc(
        config_.memory.alignmentSize, total_size * sizeof(StorageType))));
    return embedding_data != nullptr;
}
```
:::

# Optimization Approaches

This chapter introduces and discusses the optimization approaches that
will be benchmarked later.

## Baseline implementation

### Naive float32 implementation

The naive and simple float implementation resides in the class
*EmbeddingSearchFloat*. In this class the embeddings vectors are set by
simply copying the vectors from the load
function [\[alg:setEmbeddingsFloat\]](#alg:setEmbeddingsFloat){reference-type="ref"
reference="alg:setEmbeddingsFloat"}. The similarities are calculated by
iterating over all embeddings and comparing the similarity for each with
the query. This stays the same for all searcher
implementations [\[alg:similarityloop\]](#alg:similarityloop){reference-type="ref"
reference="alg:similarityloop"}. The cosine similarity for two vectors
is calculated by iterating over every vector element. Then the
corresponding vector elements of the input vectors are multiplied to
each other. The result is then added to the dot product. After the loop
we have the correct dot product value in the value *dot_product*. The
cosine similarity is then derived from the dot product like its
explained in [2.1.2.2](#sec:cosinetodot){reference-type="ref"
reference="sec:cosinetodot"}. After all similarities are calculated in
[\[alg:similarityloop\]](#alg:similarityloop){reference-type="ref"
reference="alg:similarityloop"} the results are sorted by the cosine
similarity and returned. Generally this version is the slowest and uses
the most memory because it doesn't use any quantization or special
instructions to safe memory or accelerate the computation.

<figure>
</figure>

<figure>
</figure>

<figure>
</figure>

## Quantization Methods

### Int8 quantization {#int8quant}

While the int8 quantization is not implemented without any other
optimization (like AVX2 or optimized memory management) it works by
multiplying each vector element by 127. The embeddings have to be
normalized for this. If they are not normalized the original values can
exceed -1 or 1 and cause the 8-bit integer to overflow. The loop
iterating over every embedding looks like the one shown in
[\[alg:similarityloop\]](#alg:similarityloop){reference-type="ref"
reference="alg:similarityloop"}. The cosine similarity is calculated by
simply calculating the dot product as seen in
[\[alg:int8cosinesimilarity\]](#alg:int8cosinesimilarity){reference-type="ref"
reference="alg:int8cosinesimilarity"}. Dividing by the magnitude of the
vectors isn't necessary, because the vectors have already been
normalized. Int8 embeddings use $\frac{1}{4}$ of the memory that the
float32 embeddings use. The cosine calculation is also faster, because
multiplying int8 values is faster than multiply float32. Another speedup
factor is the vector normalization before calculating the cosine
similarity. One downside is that the multiplication results of the two
vector elements have to be stored in an int16 value because in the worst
case you need 16bit for the result of an 8-bit int multiplication.

<figure>
</figure>

### Binary quantization {#binaryquant}

For the binary quantization the conversion from the float value to
binary is very simple. Because one vector element has to be represented
by one bit it represents negative values as `0` and positive values as
`1`. Instead of just creating a vector of booleans 64 vector elements
are represented by one 64bit int value. Now the binary embeddings have
$\frac{1}{64}$ the original vector size. The algorithm that does the
conversion is shown in
[\[alg:binaryquant\]](#alg:binaryquant){reference-type="ref"
reference="alg:binaryquant"}. Here the vectors don't need to be
normalized, because a binary vector can have just one length anyway.
Again the function iterating over every embedding and calculating the
cosine similarity is like the one shown in
[\[alg:similarityloop\]](#alg:similarityloop){reference-type="ref"
reference="alg:similarityloop"}. Then we can use XNOR-based binary
multiplication to build the dot product. The whole 64bit integer of the
query gets XNORd with the integer from the embedding. Then popcount is
used to count the ones as ones represent the matching bits. This is
shown in [\[alg:binarycosine\]](#alg:binarycosine){reference-type="ref"
reference="alg:binarycosine"}. The higher the returned dot product is
the more similar the vectors are.

<figure>
</figure>

<figure>
</figure>

## Dimension reduction

## SIMD optimization

### Float32 implementation with AVX2 {#floatimplavx2}

This implementation will have the same accuracy and memory usage as the
naive float implementation but it will be faster because AVX2 can
multiply two vectors, consisting of 8 floats each, at once. We convert
the default float vectors into AVX2 by simply loading the original
vectors in 256 bit / 32 Byte steps into the AVX2 Vector as shown in
[\[alg:floatvectoavx2\]](#alg:floatvectoavx2){reference-type="ref"
reference="alg:floatvectoavx2"}.

Computing the cosine similarity works generally like in the naive float
version but with AVX2 instructions. The computation will be explained
with the algorithm in
[\[alg:cosinefloatavx2\]](#alg:cosinefloatavx2){reference-type="ref"
reference="alg:cosinefloatavx2"}. Line 12-14 initializes the running
sums for the dot product and magnitudes. The for loop iterates over the
whole vector where each element is actually an AVX2 vector with 8 float
elements. Next the two vectors are multiplied. After that the product is
added to the dot product. Line 18, 19 squares each vector and then adds
the result to the magnitude for the corresponding vector. After the for
loop the sum of all float values in the three AVX2 vectors
(*dot_product*, *mag_a*, *mag_b*), also called horizontal sum, are
calculated with the helper function *\_mm256_reduce_add_ps*. \[TODO:
Explain the horizontal add function too?\]

<figure>
</figure>

<figure>
</figure>

### Int8 quantization with AVX2 {#int8implavx2}

The conversion from float to int8 works like explained in section
[4.2.1](#int8quant){reference-type="ref" reference="int8quant"}. But
this time we can store 32 int8 values in one AVX2 vector. \[TODO explain
algorithm
[\[alg:cosineint8avx2\]](#alg:cosineint8avx2){reference-type="ref"
reference="alg:cosineint8avx2"}\]

<figure>
</figure>

### Binary quantization with AVX2 {#binaryquantavx2}

Just like in [4.2.2](#binaryquant){reference-type="ref"
reference="binaryquant"} we just store the sign of the float embeddings.
This time we can store 256 binary values in one AVX2 vector. Cosine
similarity is computed by using XOR on both 256 bit vectors and then
using XOR with a vector full of ones on the result. This flips every bit
of the result. After that we have the XNOR result. We have to use
*popcountll* four times because each can only build the popcount of 64
bits. See
[\[alg:cosinebinaryavx2\]](#alg:cosinebinaryavx2){reference-type="ref"
reference="alg:cosinebinaryavx2"}.

<figure>
</figure>

### Optimized float32 implementation with AVX2 {#floatoavx2}

Like explained in [3.1.2.1](#memorymanagement){reference-type="ref"
reference="memorymanagement"} the optimized implementation uses manually
managed memory to store the embeddings. The embeddings are set using
memcpy
[\[alg:loadingembeddingsoptavx2\]](#alg:loadingembeddingsoptavx2){reference-type="ref"
reference="alg:loadingembeddingsoptavx2"}. The cosine similarity works
different (see
[\[alg:cosinefloatoavx2\]](#alg:cosinefloatoavx2){reference-type="ref"
reference="alg:cosinefloatoavx2"}): Here the similarity function only
gets a pointer to the vectors. Furthermore, the function expects both
vectors to be normalized already. This way it doesn't have to calculate
that magnitudes of the vectors, which also saves some computation time.
In addition to this loop unrolling is used. One loop iteration works
through 8 vectors instead of one. Although modern CPUs and compilers can
unroll loops on their own it is implemented it manually here. The unroll
factor is 8. That means 16 AVX vectors are loaded per iteration, which
makes perfect use of the 16 256-bit AVX registers [@intel64manual].
Beyond that line 20-23 use a prefetching instruction to preload vectors
into cache that will be accessed within a few loop iterations. This can
also help, especially if it's fine-tuned for the used hardware
[@prefetching]. Finally, this implementation also uses fused
multiply-add, which has the benefit of being faster, more accurate due
to less rounding and being more predictable [@fma]. At last the
horizontal sum is calculated the same way the unoptimized AVX2 version
does [4.4.1](#floatimplavx2){reference-type="ref"
reference="floatimplavx2"}.

<figure>
</figure>

<figure>
</figure>

### Optimized int8 implementation with AVX2

### Optimized binary implementation with AVX2

The embeddings are set similarly to method described in
[4.2.2](#binaryquant){reference-type="ref" reference="binaryquant"} and
[4.4.3](#binaryquantavx2){reference-type="ref"
reference="binaryquantavx2"}. Like the optimized float implementation
[4.4.4](#floatoavx2){reference-type="ref" reference="floatoavx2"} this
cosine computation also uses loop unrolling and manual prefetching. The
XNOR based binary multiplication also matches the previous one
[4.4.3](#binaryquantavx2){reference-type="ref"
reference="binaryquantavx2"}. But a big improvement is the usage of
256-bit popcount algorithm implemented with AVX2 instructions. Which was
developed by @Mu_a_2017 [@Mu_a_2017]. Cosine similarity algorithm can be
seen here
[\[alg:cosinebinaryoavx2\]](#alg:cosinebinaryoavx2){reference-type="ref"
reference="alg:cosinebinaryoavx2"}.

<figure>
</figure>

## Two-step search with rescoring

The two-step approach uses a fast but lower accuracy search first to
filter for candidates. Then a full accuracy search is done on these
candidates. The top results from the second search are final result. The
goal of this method is to combine the speed of the lower accuracy search
with the high accuracy of full precision search. The amount of documents
the first search retrieves can be tuned with the rescoring factor. The
two-step searcher is implemented using the optimized versions of the
binary and float32 searchers displayed in .

<figure>
</figure>

# Experimental evaluation

## Methodology

### Dataset

The dataset which will be used for benchmarking the optimization methods
described in the previous chapters is the Wikipedia article dataset from
Wikimedia available on hugging face[^1]. This set contains over 6
million English articles from Wikipedia. Instead of encoding all
articles, which would result in almost 23 GB in vector embeddings, 1.2
million randomly selected articles were encoded resulting in 4.6 GB of
embeddings for 1024 dimensions. The data was converted by two different
embedding models: The first one is the model which is a very good
performing model on the *Massive Text Embedding Benchmark* (available
here[^2]) paper [@muennighoff2023mtebmassivetextembedding]. This is a
vector angle optimized model [@li2024angleoptimizedtextembeddings] which
should give binary quantization an advantage compared to traditional
models. Additionally, this model embeds to 1024 vector dimensions and
has 335M parameters. To have a comparison the dataset of 1.2 million
articles will be encoded with the model available here[^3]. This model
maps the encoded text to 768 dimensions and has 109M parameters. It
scores a bit worse on the *Massive Text Embedding Benchmark* but is
still a solid model (rank 117 of 476 for the sentence-transformer model
compared to 35 of 476 for the mixedbread-ai model as of 2024-11-29).

To benchmark the different search methods a list containing 303 queries
will be used. It contains long and short-, specific and unspecific-,
pointless queries,\...

### Evaluation metrics

To evaluate the benchmark results the Jaccard index and NDCG will be
used to compare the performance to the naive float implementation as a
baseline. Additionally, the time taken for each search will be measured.
This includes initialization of the results array, calculating the
similarity for every embedding, sorting for top k results and returning
the result. At last the theoretical memory usage will be calculated, see.


### Testing environment

All benchmarks were recorded on Linux using kernel version 6.11.
Specifically version shipped by fedora. The test system has an AMD 5950X
CPU (16C/32T) with disabled turbo boost to make it run consistently at
3.4GHz. Furthermore, 128GB of dual-channel/quad-rank DDR4 SDRAM running
at 3600MTs/1800MHz with CL18 were used.

## Results and Analysis

### Search Time comparison

Starting with the figure : As expected the AVX2 optimized version is
much faster than the naive implementation. The fact it's over 8 times
faster, taking only 133.5ms compared to 1100ms for float, shows that
this implementation makes efficient use of the AVX2 instructions.

The binary implementation, which also uses AVX2, again, is over 23 times
faster than the AVX2 float equivalent. Which makes sense, because it
only has to iterate through $1/32$th of the data and uses efficient nor,
xor and popcount instructions.

With 95ms The int8 implementation is slower than expected. But that is
likely due to the fact that AVX2 can't multiply 32 int8 values at once.
It has to extract the lower/higher 128 bits, sign extend them to 16bit
and then do 2 multiplications for the high and low values. After that it
has to build the horizontal sum and add the high and low multiplication
result to the running sum. This is a significant amount of instructions
compared to the AVX2 float implementation where 32 floats are multiplied
and added to the running sum by just 4 FMA instructions.

The PCA variants perform quiet well. The dimension reduction to half
almost doubles the search speed compared to the float AVX2
implementation, which makes sense, given the fact that the vectors only
have half the dimensions. The other PCA factors almost scale linearly as
well. PCA4 is more than 2 times faster than int8 which has to go through
the same amount of data. This shows that AVX2 doesn't really perform
well at multiplying int8 values.

The two-step method based on the binary and float AVX2 implementation is
very close to the pure binary search time. Even for high numbers of
retrieved documents and rescoring factors, the amount of embeddings the
second step method has to search is very small, e.g. 5000 for k=100
(like in ) and rescoring factor of 50, compared to the total number of
embeddings.

With 540ms the mapped float searcher has the second-worst search time
after the naive float. The reason for this is, that the CPU prefetcher
can only predict the int8 values that will be loaded next. But after
this gather instructions are used to load the corresponding float values
from the mapping table. Even though it's very small (1 kB) and likely
stays in L1 cache the entire time, the very low L1 cache latency of
around 1ns adds up: Taking the benchmarked dataset with 1.2M embeddings
with 1024 dimensions it would take
$10^{-9}s * 1024 * 1.2 * 10^{6} / 8 = 153.6ms$ just to load the float
values for the embeddings. And this makes the assumption, that loading 8
float values at once takes 1ns. In conclusion, this methods' reliance on
random memory access slows the search significantly. This is also seen
in the low memory bandwidth in
[\[memorybandwidth\]](#memorybandwidth){reference-type="ref"
reference="memorybandwidth"}.

Finally, the two-step method using binary and the mapped float performs
similar to binary just like the two-step method that uses full accuracy
search in the second step. Only as the rescoring factor increases it
takes slightly longer than its competitor. That happens, because the
mapped float searcher is much slower than the avx2 searcher and as the
rescoring factor increases the speed of the second-step searcher becomes
more relevant. Nonetheless, this search method performs really well, as
long the rescoring factors are reasonable.

<figure>
</figure>

### Accuracy Analysis

The AVX2 implementation has perfect accuracy as seen in
[\[accuracyheatmapone\]](#accuracyheatmapone){reference-type="ref"
reference="accuracyheatmapone"}. But one can spot one outlier for the
Jaccard index and NDCG score in and respectively. But that is likely due
to the fact that the AVX2 implementation can actually be more accurate
with the fused multiply add instruction, which eliminates one rounding
step.

<figure>
</figure>

<figure>
</figure>

The binary searcher performs decent with an average NDCG of 0.576 and
Jaccard index of 0.438 when considering its heavy quantization and the
resulting fast search time and low memory usage. The PCA32 method, for
example, uses the same amount of memory but has a much lower score which
makes it useless when comparing it with the binary searcher. The NDCG
being higher than the Jaccard index indicates that the binary searcher
also retrieves the important documents quiet reliably. In the box plots
in and
[\[boxndcgsearchersone\]](#boxndcgsearchersone){reference-type="ref"
reference="boxndcgsearchersone"} show some outliers close to, and below
0.2 Jaccard index and slightly above 0.2 NDCG. This means that for some
queries it can be unreliable on its own.

The int8 based searcher performs well with an average NDCG of 0.9 and
Jaccard index of 0.87. It performs exclusively above the binary
searcher. The outliers are also a lot less drastic. Especially the NDCG
stays above 0.8 for all queries except one.

Float16 or half precision float has very good average accuracy scores .
The box plots in
[\[boxjaccardsearchersone\]](#boxjaccardsearchersone){reference-type="ref"
reference="boxjaccardsearchersone"} and also show it performing very
good. Especially the first and third quartile have a very close range
and stay above 0.92 Jaccard index and 0.94 NDCG. The outliers are also
still very good with at least 0.8 Jaccard index and 0.87 NDCG. This
makes it suitable to fully replace the float32 quantization with float16
as it gives good memory savings, reliable results and is faster on
supported hardware.

Even better performs the mapped float searcher. On average, it gets
close to perfect scores (). The outliers are no worse than the float16
outliers and the lower percentiles are above 0.96 for both metrics. This
makes it the most accurate quantization tested.

The variants with PCA dimension reduction applied to the embeddings
mostly perform worse than methods with the similar memory usage and
search speed. Int8 is more accurate than PCA2 and only uses half the
memory. While PCA2 only got a slight speed advantage. The bad
performance when using PCA for embeddings is also mentioned here
[@thakur2023injectingdomainadaptationlearningtohash].

<figure>
</figure>

Both two-step methods (binary+float and binary+mapped float) perform
really well. Especially with rescoring factors of 10 and higher. For a
rescoring factor of 10 they have a first quartile NDCG score of 0.96 and
0.97 respectively. The outliers from the binary searcher still exists
and only get slightly better as the rescoring factor increases. With a
rescoring factor of 25 or higher the binary+float searcher mostly gets
perfect scores, except for the outliers. The score for the binary+mapped
float searcher is capped by the mapped float searcher which is still
very high.

### Accuracy vs Rescoring Factor

As mentioned in the previous section the search time increases when
increasing the rescoring factor. The increase in time is linear as seen
in and . The search time overall stays very low as even with high
rescore factors the number of prefiltered embeddings is very small
compared to the number of total embeddings. At a rescoring factor of 25
the score is very close to the full search equivalent of the second
method. The only point in increasing it further is to reduce outliers.

<figure>
</figure>

### Comparing Benchmark Results from Different Models

As mentioned earlier the model used is an angle optimized model, which
should enable the binary searcher to get better results
[@emb2024mxbai; @li2024angleoptimizedtextembeddings]. To verify the
previous results, the scores of the searchers when using the model will
be compared against the scores when using the text embedding model. The
same queries will be used.

Looking at the bar plot
[\[ndcg_comp_diff_models\]](#ndcg_comp_diff_models){reference-type="ref"
reference="ndcg_comp_diff_models"} we see that the binary searcher
indeed performs better with the angle optimized model. So does the
two-step search based on the binary searcher as the first step. The PCA
reduced searcher also perform better. There is no significant difference
for the other searchers.

<figure>
</figure>

### Memory Usage Analysis

<figure>
</figure>

[^1]: <https://huggingface.co/datasets/wikimedia/wikipedia>

[^2]: <https://huggingface.co/spaces/mteb/leaderboard>

[^3]: <https://huggingface.co/sentence-transformers/all-mpnet-base-v2>
