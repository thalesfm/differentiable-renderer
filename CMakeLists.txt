cmake_minimum_required(VERSION 3.5)

project(drt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(render src/render.cpp)
target_include_directories(render PRIVATE include ext/tclap/include)
target_link_libraries(render PRIVATE m Half IlmImf)
target_compile_options(render PRIVATE "$<$<CONFIG:Debug>:-Og;-ggdb;-Wall;-Wpedantic>")
target_compile_options(render PRIVATE "$<$<CONFIG:Release>:-O3>")

add_subdirectory(ext/openexr EXCLUDE_FROM_ALL)
