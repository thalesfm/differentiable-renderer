cmake_minimum_required(VERSION 3.5)

project(drt-adjoint)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(render src/render.cpp)
target_include_directories(render PRIVATE include)
target_link_libraries(render PRIVATE m armadillo Half IlmImf)
target_compile_options(render PRIVATE -Wall -Wpedantic)
target_compile_options(render PRIVATE "$<$<CONFIG:Debug>:-Og -ggdb>")
target_compile_options(render PRIVATE "$<$<CONFIG:Release>:-O3>")
