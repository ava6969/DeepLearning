install(FILES recurrent_net.h recurrent_net.cpp DESTINATION include/${PROJECT_NAME}/memory)

target_sources(${PROJECT_NAME} PRIVATE recurrent_net.cpp)