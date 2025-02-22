cmake_minimum_required(VERSION 3.15)
#include(build/Release/generators/conan_toolchain.cmake)
project(simsearch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "include-what-you-use")

add_compile_options(-Wno-ignored-attributes)

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# Add the include directory to the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# main executable (main.cpp)
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/embedding_io.cpp
    src/embedding_search_avx2.cpp
    src/embedding_search_binary.cpp
    src/embedding_search_float.cpp
    src/embedding_search_binary_avx2.cpp
    src/embedding_search_uint8_avx2.cpp
    src/embedding_utils.cpp
    src/optimized_embedding_search_uint8_avx2.cpp
    src/optimized_embedding_search_avx2.cpp
    src/optimized_embedding_search_binary_avx2.cpp
    src/embedding_search_float16.cpp
    src/embedding_search_mapped_float.cpp
)

# Create the Python module
pybind11_add_module(embedding_search_benchmark
    src/embedding_search_benchmark_bindings.cpp
    src/embedding_io.cpp
    src/embedding_search_avx2.cpp
    src/embedding_search_binary.cpp
    src/embedding_search_float.cpp
    src/embedding_search_binary_avx2.cpp
    src/embedding_search_uint8_avx2.cpp
    src/embedding_utils.cpp
    src/optimized_embedding_search_uint8_avx2.cpp
    src/optimized_embedding_search_avx2.cpp
    src/optimized_embedding_search_binary_avx2.cpp
    src/embedding_search_float16.cpp
    src/embedding_search_mapped_float.cpp
)

# Link libraries for main executable
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        arrow
        parquet
)

# Link libraries for Python module
target_link_libraries(embedding_search_benchmark
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        arrow
        parquet
)

# Set Python module properties
set_target_properties(embedding_search_benchmark PROPERTIES
    PREFIX ""  # Removes lib prefix on Unix
    SUFFIX ".so"  # Sets correct extension on Unix
)

# set different flags for Debug and Release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -mavx2 -DNO_THREADS")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -fopenmp -mavx2 -DNO_THREADS")