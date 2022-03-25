install(FILES recurrent_net.h DESTINATION include/${PROJECT_NAME}/memory)

target_sources(${PROJECT_NAME} PRIVATE recurrent_net.tpp)