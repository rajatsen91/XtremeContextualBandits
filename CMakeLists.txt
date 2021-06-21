cmake_minimum_required(VERSION 2.8.12)
project(xcb)
# Set source directory
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCE_DIR "src/xcb")
# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR}/corelib)
set(SOURCES "${SOURCE_DIR}/corelib/toy.cpp" "${SOURCE_DIR}/corelib/xcb_inference.cpp")
add_subdirectory(lib/pybind11)
include_directories(lib/eigen)
pybind11_add_module(core ${SOURCES} "${SOURCE_DIR}/corelib/bindings.cpp")