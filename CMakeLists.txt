cmake_minimum_required(VERSION 3.14)
project(MtrxSLAE CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GoogleTest через FetchContent
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP true
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Основная программа
add_executable(MtrxSLAE_main main.cpp)
target_include_directories(MtrxSLAE_main PRIVATE ${CMAKE_SOURCE_DIR})

# Тесты
add_executable(MtrxSLAE_tests
        tests/test_DynamicArray.cpp
        tests/test_LinkedList.cpp
        tests/test_Sequence.cpp
        tests/test_MapReduce.cpp
        tests/test_Matrix.cpp
        tests/test_Solver.cpp
)
target_include_directories(MtrxSLAE_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(MtrxSLAE_tests GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(MtrxSLAE_tests)
