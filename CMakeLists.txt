cmake_minimum_required(VERSION 3.4)
project(orb_slam3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(./)

find_package(Python 3 REQUIRED COMPONENTS Interpreter Development NumPy)
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

file(GLOB_RECURSE ORB_SLAM3_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)
pybind11_add_module(orb_slam3 MODULE ${ORB_SLAM3_SRC})

target_link_libraries(orb_slam3 PRIVATE ${OpenCV_LIBS})
target_include_directories(orb_slam3 PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(orb_slam3 PRIVATE ${PYTHON_LIBRARIES})
target_include_directories(orb_slam3 PRIVATE ${PYTHON_INCLUDE_DIRS})
