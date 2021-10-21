cmake_minimum_required(VERSION 3.0.0)
project(lightgbm_predict VERSION 0.2.0)

set(SOURCE_CODE
    ./main.cc)

add_executable(lightgbm_predict ./main.cc)

if(LIGHTGBM_INC)
    if(EXISTS "${LIGHTGBM_INC}/LightGBM/c_api.h")
        message(STATUS "Found ${LIGHTGBM_INC}/LightGBM/c_api.h in LIGHTGBM_INC dir")
    else()
        message(SEND_ERROR "Could not find ${LIGHTGBM_INC}/LightGBM/c_api.h in LIGHTGBM_INC dir")
    endif()
    include_directories("${LIGHTGBM_INC}")
endif()

if(LIGHTGBM_LIB)
    find_library(LIGHTGBM_LIBRARIES lightgbm PATHS "${LIGHTGBM_LIB}" REQUIRED)
    message(STATUS "Found ${LIGHTGBM_LIBRARIES}")
    target_link_libraries(lightgbm_predict "${LIGHTGBM_LIB}/lib_lightgbm.lib")
else()
    message(SEND_ERROR "You need to provide -DLIGHTGBM_LIB=... to cmake.")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
