cmake_minimum_required(VERSION 3.17)

set(CMAKE_TOOLCHAIN_FILE $ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
option(BUILD_EXAMPLES ON)

project(deep_networks)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch)
add_subdirectory(tensorboard_logger)

find_package(Torch REQUIRED)
find_package(yaml-cpp  REQUIRED)

add_library(${PROJECT_NAME} STATIC "" vision/impala_residual_block.cpp)
target_include_directories( ${PROJECT_NAME} PRIVATE .)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} yaml-cpp tensorboard_logger)

add_subdirectory(basic)
add_subdirectory(vision)
add_subdirectory(memory)
add_subdirectory(common)
add_subdirectory(distribution)


if(${BUILD_EXAMPLES})
    file(GLOB Examples examples/*.cpp)
    foreach(SRC_FILE IN LISTS Examples)
        string(REPLACE .cpp "" executable_name ${SRC_FILE})
        string(REPLACE ${CMAKE_SOURCE_DIR}/examples/ "" executable_name ${executable_name})
        add_executable(${executable_name} ${SRC_FILE} vision/impala_residual_block.h vision_debugger.h)
        target_link_libraries(${executable_name} PRIVATE ${PROJECT_NAME} ${TORCH_LIBRARIES} yaml-cpp tensorboard_logger)
        target_include_directories( ${executable_name} PRIVATE . tensorboard_logger)
    endforeach()
endif()