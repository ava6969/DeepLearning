set(CMAKE_TOOLCHAIN_FILE $ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

project(tensorboard_logger)
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set (CMAKE_CXX_STANDARD 11)
find_package(protobuf CONFIG REQUIRED)

message('protofbuf ${Protobuf_VERSION}')
file(GLOB protos "proto/*.proto")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${protos})

add_library(tensorboard_logger SHARED
    "src/crc.cc"
    "src/tensorboard_logger.cc"
    ${PROTO_SRCS}
)
target_include_directories(tensorboard_logger PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${Protobuf_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}
    )
target_link_libraries(tensorboard_logger PUBLIC protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

add_executable(test
    tests/test_tensorboard_logger.cc
)
target_link_libraries(test tensorboard_logger)
