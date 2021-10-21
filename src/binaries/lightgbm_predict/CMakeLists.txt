cmake_minimum_required(VERSION 3.0.0)
project(lightgbm_predict VERSION 0.1.0)

include(CTest)
enable_testing()

set(SOURCE_CODE
    ./main.cc)

add_subdirectory("${PROJECT_SOURCE_DIR}/LightGBM")
include_directories("${PROJECT_SOURCE_DIR}/LightGBM/include")

add_executable(lightgbm_predict ./main.cc)
target_link_libraries(lightgbm_predict _lightgbm)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
