cmake_minimum_required(VERSION 3.0.0)
project(lightgbm_benchmark_utils VERSION 0.3.0)

set(SOURCE_CODE
    ./lightgbm_predict/main.cc)

add_executable(lightgbm_predict ./lightgbm_predict/main.cc)

if(LIGHTGBM_LIB)
    message(STATUS "Looking for lib_lightgbm at ${LIGHTGBM_LIB}")
    find_library(LIGHTGBM_LIBRARIES NAMES _lightgbm lib_lightgbm PATHS "${LIGHTGBM_LIB}" REQUIRED)
else()
    message(STATUS "Looking for lib_lightgbm in PATH(win) or LD_LIBRARY_PATH(linux)")
    find_library(LIGHTGBM_LIBRARIES NAMES _lightgbm lib_lightgbm REQUIRED)
endif()

if(NOT LIGHTGBM_LIBRARIES)
    message(SEND_ERROR "We could not find lib_lightgbm. Please provide -DLIGHTGBM_LIB=... to cmake command")
else()
    message(STATUS "Found lib_lightgbm at ${LIGHTGBM_LIBRARIES}")
    target_link_libraries(lightgbm_predict "${LIGHTGBM_LIBRARIES}")
endif()

if(LIGHTGBM_INC)
    if(EXISTS "${LIGHTGBM_INC}/LightGBM/c_api.h")
        message(STATUS "Found LightGBM/c_api.h in ${LIGHTGBM_INC} dir")
    else()
        message(SEND_ERROR "Could not find LightGBM/c_api.h in ${LIGHTGBM_INC} dir")
    endif()
    include_directories("${LIGHTGBM_INC}")
else()
    message(WARNING "If you don't provide -DLIGHTGBM_INC=... you better hope LightGBM/c_api.h is in your includes dir.")
endif()
