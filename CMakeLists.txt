cmake_minimum_required(VERSION 3.17)

set(CMAKE_TOOLCHAIN_FILE $ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
option(BUILD_EXAMPLES ON)

project(deep_networks)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch)
set(SAM_PATH $ENV{HOME}/sam)
set(Boost_USE_STATIC_LIBS   ON)

add_subdirectory(${SAM_PATH}/tensorboard_logger ${CMAKE_BINARY_DIR}/tensorboard)

find_package(Torch REQUIRED 1.10)
find_package(tabulate REQUIRED)
find_package(yaml-cpp  REQUIRED 0.7.0)

add_library(${PROJECT_NAME} STATIC base.tpp)
target_include_directories( ${PROJECT_NAME} PRIVATE . ${SAM_PATH}/tensorboard_logger/include)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} yaml-cpp tensorboard_logger )

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
    target_link_libraries(${executable_name} PRIVATE ${PROJECT_NAME} ${TORCH_LIBRARIES} yaml-cpp )
    target_include_directories( ${executable_name} PRIVATE . )
endforeach()
endif()

function(print_target_properties tgt)
    if(NOT TARGET ${tgt})
        message("There is no target named '${tgt}'")
        return()
    endif()

    # this list of properties can be extended as needed
    set(CMAKE_PROPERTY_LIST SOURCE_DIR BINARY_DIR COMPILE_DEFINITIONS
            COMPILE_OPTIONS INCLUDE_DIRECTORIES LINK_LIBRARIES)

    message("Configuration for target ${tgt}")

    foreach (prop ${CMAKE_PROPERTY_LIST})
        get_property(propval TARGET ${tgt} PROPERTY ${prop} SET)
        if (propval)
            get_target_property(propval ${tgt} ${prop})
            message (STATUS "${prop} = ${propval}")
        endif()
    endforeach(prop)

endfunction(print_target_properties)

print_target_properties(deep_networks)