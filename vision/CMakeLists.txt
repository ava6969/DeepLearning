install(FILES conv_net.h residual_block.h DESTINATION include/${PROJECT_NAME}/vision)

target_sources(${PROJECT_NAME} PRIVATE conv_net.cpp residual_block.cpp convLSTM.cpp
        impala_residual_block.cpp relational_module.cpp)