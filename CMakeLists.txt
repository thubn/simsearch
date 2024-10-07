cmake_minimum_required(VERSION 3.15)
project(simsearch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check for AVX2 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    add_definitions(-DAVX2_ENABLED)
else()
    message(WARNING "AVX2 is not supported on this machine.")
endif()

find_package(nlohmann_json REQUIRED)

# Add the include directory to the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/embedding_io.cpp
    src/embedding_search_avx2.cpp
    src/embedding_search_binary.cpp
    src/embedding_search_float.cpp
    src/embedding_search_binary_avx2.cpp
)
# Link the library and any other dependencies
target_link_libraries(${PROJECT_NAME}
    PRIVATE 
        nlohmann_json::nlohmann_json
)

# Optionally, you can set different flags for Debug and Release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Print include directories for debugging
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()