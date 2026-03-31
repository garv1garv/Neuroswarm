cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(NeuroSwarm VERSION 1.0.0 LANGUAGES CXX CUDA)

# ──────────────────────────────────────────────────────────
# Build Configuration
# ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmark suite" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" ON)
option(ENABLE_TENSORRT "Enable TensorRT-LLM integration" ON)
option(ENABLE_PROFILING "Enable CUDA profiling hooks" OFF)
option(ENABLE_SANITIZERS "Enable address/thread sanitizers" OFF)

# ──────────────────────────────────────────────────────────
# CUDA Architecture Detection
# ──────────────────────────────────────────────────────────
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Target Ampere (80), Ada Lovelace (89), Hopper (90)
    set(CMAKE_CUDA_ARCHITECTURES "80;89;90")
endif()

# ──────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# TensorRT
if(ENABLE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64)
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
        set(HAS_TENSORRT TRUE)
    else()
        message(WARNING "TensorRT not found. Building without TRT support.")
        set(HAS_TENSORRT FALSE)
    endif()
endif()

# Redis (hiredis)
find_library(HIREDIS_LIBRARY hiredis)
find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)

# pybind11
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.12.0)
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# Google Test
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# spdlog for logging
include(FetchContent)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0)
FetchContent_MakeAvailable(spdlog)

# nlohmann/json
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

# Prometheus-cpp (metrics)
FetchContent_Declare(prometheus_cpp
    GIT_REPOSITORY https://github.com/jupp0r/prometheus-cpp.git
    GIT_TAG v1.2.4)
set(ENABLE_PUSH OFF CACHE BOOL "" FORCE)
set(ENABLE_COMPRESSION OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(prometheus_cpp)

# ──────────────────────────────────────────────────────────
# Compiler Flags
# ──────────────────────────────────────────────────────────
set(NEUROSWARM_CXX_FLAGS
    -Wall -Wextra -Wpedantic -Wno-unused-parameter
    -march=native -O3 -ffast-math
)
set(NEUROSWARM_CUDA_FLAGS
    --use_fast_math
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler=-fPIC
    -lineinfo
)

if(ENABLE_PROFILING)
    list(APPEND NEUROSWARM_CUDA_FLAGS -DNEUROSWARM_PROFILING)
    list(APPEND NEUROSWARM_CXX_FLAGS -DNEUROSWARM_PROFILING)
endif()

if(ENABLE_SANITIZERS)
    list(APPEND NEUROSWARM_CXX_FLAGS -fsanitize=address -fsanitize=undefined)
endif()

# ──────────────────────────────────────────────────────────
# CUDA Kernel Library
# ──────────────────────────────────────────────────────────
add_library(neuroswarm_cuda STATIC
    cuda/attention_kernels.cu
    cuda/memory_pool.cu
    cuda/quantization_kernels.cu
    cuda/rope_embeddings.cu
    cuda/comm_kernels.cu
)
target_compile_options(neuroswarm_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NEUROSWARM_CUDA_FLAGS}>
)
target_include_directories(neuroswarm_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(neuroswarm_cuda PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

# ──────────────────────────────────────────────────────────
# Core C++ Library
# ──────────────────────────────────────────────────────────
add_library(neuroswarm_core STATIC
    src/orchestrator.cpp
    src/agent_runtime.cpp
    src/communication.cpp
    src/memory_manager.cpp
    src/inference_engine.cpp
    src/circuit_breaker.cpp
    src/metrics_collector.cpp
    src/sandbox.cpp
)
target_compile_options(neuroswarm_core PRIVATE ${NEUROSWARM_CXX_FLAGS})
target_include_directories(neuroswarm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(neuroswarm_core PUBLIC
    neuroswarm_cuda
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    prometheus-cpp::pull
    Threads::Threads
    CUDA::cudart
)

if(HAS_TENSORRT)
    target_include_directories(neuroswarm_core PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(neuroswarm_core PUBLIC ${TENSORRT_LIBRARY})
    target_compile_definitions(neuroswarm_core PUBLIC HAS_TENSORRT)
endif()

if(HIREDIS_LIBRARY)
    target_include_directories(neuroswarm_core PUBLIC ${HIREDIS_INCLUDE_DIR})
    target_link_libraries(neuroswarm_core PUBLIC ${HIREDIS_LIBRARY})
    target_compile_definitions(neuroswarm_core PUBLIC HAS_REDIS)
endif()

# ──────────────────────────────────────────────────────────
# Main Executable
# ──────────────────────────────────────────────────────────
add_executable(neuroswarm src/main.cpp)
target_link_libraries(neuroswarm PRIVATE neuroswarm_core)
target_compile_options(neuroswarm PRIVATE ${NEUROSWARM_CXX_FLAGS})

# ──────────────────────────────────────────────────────────
# Python Bindings
# ──────────────────────────────────────────────────────────
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(_neuroswarm_cpp python/bindings.cpp)
    target_link_libraries(_neuroswarm_cpp PRIVATE neuroswarm_core)
    target_compile_options(_neuroswarm_cpp PRIVATE ${NEUROSWARM_CXX_FLAGS})
endif()

# ──────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────
if(BUILD_TESTS)
    add_executable(neuroswarm_tests
        tests/test_attention_kernels.cpp
        tests/test_memory_pool.cpp
        tests/test_orchestrator.cpp
        tests/test_communication.cpp
        tests/test_agent_runtime.cpp
        tests/test_circuit_breaker.cpp
        tests/test_quantization.cpp
        tests/test_integration.cpp
    )
    target_link_libraries(neuroswarm_tests PRIVATE
        neuroswarm_core
        GTest::gtest_main
    )
    target_compile_options(neuroswarm_tests PRIVATE ${NEUROSWARM_CXX_FLAGS})
    include(GoogleTest)
    gtest_discover_tests(neuroswarm_tests)
endif()

# ──────────────────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────────────────
if(BUILD_BENCHMARKS)
    add_executable(bench_attention cuda/benchmarks/bench_attention.cu)
    target_link_libraries(bench_attention PRIVATE neuroswarm_cuda)
    target_compile_options(bench_attention PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${NEUROSWARM_CUDA_FLAGS}>
    )

    add_executable(bench_inference cuda/benchmarks/bench_inference.cu)
    target_link_libraries(bench_inference PRIVATE neuroswarm_core)
    target_compile_options(bench_inference PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${NEUROSWARM_CUDA_FLAGS}>
    )

    add_executable(bench_communication cuda/benchmarks/bench_communication.cu)
    target_link_libraries(bench_communication PRIVATE neuroswarm_core)
    target_compile_options(bench_communication PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${NEUROSWARM_CUDA_FLAGS}>
    )
endif()

# ──────────────────────────────────────────────────────────
# Install
# ──────────────────────────────────────────────────────────
install(TARGETS neuroswarm neuroswarm_core neuroswarm_cuda
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY agents/ DESTINATION share/neuroswarm/agents)

message(STATUS "========================================")
message(STATUS " NeuroSwarm v${PROJECT_VERSION}")
message(STATUS " CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS " TensorRT:          ${HAS_TENSORRT}")
message(STATUS " Redis:             ${HIREDIS_LIBRARY}")
message(STATUS " Tests:             ${BUILD_TESTS}")
message(STATUS " Benchmarks:        ${BUILD_BENCHMARKS}")
message(STATUS " Python Bindings:   ${BUILD_PYTHON_BINDINGS}")
message(STATUS " Profiling:         ${ENABLE_PROFILING}")
message(STATUS "========================================")
