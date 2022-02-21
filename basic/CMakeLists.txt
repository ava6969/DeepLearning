install(FILES fcnn.h DESTINATION include/${PROJECT_NAME}/basic)

target_sources(${PROJECT_NAME} PRIVATE fcnn.cpp)