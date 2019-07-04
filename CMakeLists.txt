project(ExtendedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(kalman_sources
    src/kalman_filter.hpp
    src/kalman_filter.cpp
    )
set(fusion_sources
    src/FusionEKF.hpp
    src/FusionEKF.cpp
    )
set(tools_sources
    src/tools.hpp
    src/tools.cpp
    )
set(all_sources
    ${kalman_sources}
    ${fusion_sources}
    ${tools_sources}
    )

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    include_directories(/usr/local/include)
    include_directories(/usr/local/opt/openssl/include)
    link_directories(/usr/local/lib)
    link_directories(/usr/local/opt/openssl/lib)
    link_directories(/usr/local/Cellar/libuv/1*/lib)
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_executable(ExtendedKF src/main.cpp ${all_sources})

enable_testing()
add_executable(tools_test src/test/tools_test.cpp ${tools_sources})
add_test(
    NAME tools
    COMMAND $<TARGET_FILE:tools_test>
    )

target_link_libraries(ExtendedKF z ssl uv uWS)
