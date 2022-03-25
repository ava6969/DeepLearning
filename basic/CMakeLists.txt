
install(FILES fcnn.h embedding.h entity_aware.h self_attention.h DESTINATION include/${PROJECT_NAME}/basic)

target_sources(${PROJECT_NAME} PRIVATE fcnn.cpp entity_aware.cpp self_attention.cpp)