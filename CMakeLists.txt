cmake_minimum_required(VERSION 3.17)

set(SAM_PATH $ENV{HOME}/sam_old CACHE STRING "Path to SAM ROOT DIR")
set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch CACHE STRING "Path to Torch ROOT DIR")
set(CMAKE_TOOLCHAIN_FILE $ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
option(BUILD_EXAMPLES OFF)

project(deep_networks)

# Create Variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fPIC")
set(Boost_USE_STATIC_LIBS ON)

# IMPORT DEPENDENCIES
add_subdirectory(${SAM_PATH}/tensorboard_logger ${CMAKE_BINARY_DIR}/tensorboard)
find_package(Torch REQUIRED)
find_package(tabulate REQUIRED)
find_package(yaml-cpp  REQUIRED 0.7.0)

# Create Targets
add_library(${PROJECT_NAME} STATIC base.tpp)

# Combine All CMakeList
add_subdirectory(basic)
add_subdirectory(vision)
add_subdirectory(memory)
add_subdirectory(common)
add_subdirectory(distribution)

if(${BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()

# Link Dependencies and add include dirs
target_include_directories( ${PROJECT_NAME} PRIVATE .
        ${SAM_PATH}/tensorboard_logger/include ${SAM_PATH}/tensorboard_logger )
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} yaml-cpp tensorboard_logger )

message('CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}')