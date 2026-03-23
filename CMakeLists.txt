cmake_minimum_required(VERSION 3.14)
project(MtrxSLAE CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# GoogleTest
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP true
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ADTLinearOrganization
FetchContent_Declare(
        sequences_lib
        GIT_REPOSITORY https://github.com/VenEraBle842/ADTLinearOrganization.git
        GIT_TAG        master
        SOURCE_SUBDIR  "_none_"
)
FetchContent_MakeAvailable(sequences_lib)

set(SEQ_SRC ${sequences_lib_SOURCE_DIR}/src)
configure_file(${SEQ_SRC}/DynamicArray.h   ${CMAKE_SOURCE_DIR}/src/DynamicArray.h   COPYONLY)
configure_file(${SEQ_SRC}/ArraySequence.h  ${CMAKE_SOURCE_DIR}/src/ArraySequence.h  COPYONLY)
configure_file(${SEQ_SRC}/LinkedList.h     ${CMAKE_SOURCE_DIR}/src/LinkedList.h     COPYONLY)
configure_file(${SEQ_SRC}/ListSequence.h   ${CMAKE_SOURCE_DIR}/src/ListSequence.h   COPYONLY)
configure_file(${SEQ_SRC}/Sequence.h       ${CMAKE_SOURCE_DIR}/src/Sequence.h       COPYONLY)
configure_file(${SEQ_SRC}/Option.h         ${CMAKE_SOURCE_DIR}/src/Option.h         COPYONLY)
configure_file(${SEQ_SRC}/IEnumerable.h    ${CMAKE_SOURCE_DIR}/src/IEnumerable.h    COPYONLY)
configure_file(${SEQ_SRC}/IEnumerator.h    ${CMAKE_SOURCE_DIR}/src/IEnumerator.h    COPYONLY)
configure_file(${SEQ_SRC}/Tuple.h          ${CMAKE_SOURCE_DIR}/src/Tuple.h          COPYONLY)

# Основная программа
add_executable(MtrxSLAE_main main.cpp)
target_include_directories(MtrxSLAE_main PRIVATE
        ${CMAKE_SOURCE_DIR}
)

# Тесты
add_executable(MtrxSLAE_tests
        tests/test_Matrix.cpp
        tests/test_Solver.cpp
)
target_include_directories(MtrxSLAE_tests PRIVATE
        ${CMAKE_SOURCE_DIR}
)
target_link_libraries(MtrxSLAE_tests GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(MtrxSLAE_tests)
