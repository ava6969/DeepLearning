
add_executable(test_pos_encode test_pos_encode.cpp)
# Link Dependencies and add include dirs
target_include_directories( test_pos_encode PRIVATE . ..
        ${SAM_PATH}/tensorboard_logger/include ${SAM_PATH}/tensorboard_logger )
target_link_libraries(test_pos_encode PRIVATE ${PROJECT_NAME} ${TORCH_LIBRARIES} yaml-cpp tensorboard_logger )

add_executable(test_relation_module test_relation_module.cpp )
# Link Dependencies and add include dirs
target_include_directories( test_relation_module PRIVATE . ..
        ${SAM_PATH}/tensorboard_logger/include ${SAM_PATH}/tensorboard_logger )
target_link_libraries(test_relation_module PRIVATE ${PROJECT_NAME} ${TORCH_LIBRARIES} yaml-cpp tensorboard_logger )
