cmake_minimum_required(VERSION 3.15)
project(simsearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json REQUIRED)

# Add the include directory to the include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create a library from the source files
add_library(embedding_search_lib src/embedding_search.cpp)

# Link nlohmann_json to the library
target_link_libraries(embedding_search_lib PRIVATE nlohmann_json::nlohmann_json)

add_executable(${PROJECT_NAME} src/main.cpp)
# Link the library and any other dependencies
target_link_libraries(${PROJECT_NAME}
    PRIVATE 
        embedding_search_lib
        nlohmann_json::nlohmann_json
)

# Print include directories for debugging
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()