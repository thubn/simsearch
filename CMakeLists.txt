cmake_minimum_required(VERSION 3.15)
include(${CMAKE_BINARY_DIR}/Release/generators/conan_toolchain.cmake)
project(simsearch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check for AVX2 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    add_definitions(-DAVX2_ENABLED)
else()
    message(WARNING "AVX2 is not supported on this machine.")
endif()

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add the include directory to the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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
)

# Create the Python module
pybind11_add_module(embedding_search 
    src/embedding_search_bindings.cpp
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
target_link_libraries(embedding_search
    PRIVATE
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        arrow
        parquet
)

# Set Python module properties
set_target_properties(embedding_search PROPERTIES
    PREFIX ""  # Removes lib prefix on Unix
    SUFFIX ".so"  # Sets correct extension on Unix
)

# Optionally, you can set different flags for Debug and Release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Print include directories for debugging
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()