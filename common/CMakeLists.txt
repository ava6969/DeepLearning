install(FILES common.h options.h sam_exceptions.h yaml_helper.h yaml_options.h builder.h summary.h
        DESTINATION include/${PROJECT_NAME}/common)

target_sources(${PROJECT_NAME} PRIVATE common.cpp builder.cpp)