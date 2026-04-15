cmake_minimum_required(VERSION 3.20)
project(aaedn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=znver2 -mavx2 -mfma -O3 -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=znver2 -O3")

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

set(SOURCES
    src/arena.cpp
    src/tensor.cpp
    src/matmul.cpp
    src/dot.cpp
    src/softmax.cpp
    src/rmsnorm.cpp
    src/thread.cpp
    src/smt.cpp
    src/bandwidth.cpp
    src/quant.cpp
    src/dequant.cpp
    src/rope.cpp
    src/kvcache.cpp
    src/attention.cpp
    src/ffn.cpp
    src/transformer.cpp
    src/quant_oracle.cpp
    src/fused_matmul.cpp
    src/layer_fusion.cpp
    src/tokenizer.cpp
    src/rag.cpp
    src/agent.cpp
    src/main.cpp
)

add_executable(aaedn ${SOURCES})

target_link_libraries(aaedn pthread)