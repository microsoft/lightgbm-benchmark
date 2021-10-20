cmake_minimum_required(VERSION 3.0.0)
project(lightgbm_predict_telemetry VERSION 0.1.0)

include(CTest)
enable_testing()

set(SOURCE_CODE
    src/binaries/lightgbm_predict_telemetry/main.cc)

add_subdirectory("${PROJECT_SOURCE_DIR}/libraries/LightGBM")
include_directories("${PROJECT_SOURCE_DIR}/libraries/LightGBM/include")

add_executable(lightgbm_predict_telemetry src/binaries/lightgbm_predict_telemetry/main.cc)
target_link_libraries(lightgbm_predict_telemetry _lightgbm)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
