cmake_minimum_required(VERSION 2.6)

project (test_caffe2_and_opencv_cpp)

find_package(Protobuf REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads)

include_directories(include)

include_directories(${EIGEN3_INCLUDE_DIR})

if(OpenCV_LIBS)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

find_library(CAFFE2_LIB caffe2)
find_library(GLOG_LIB glog)
find_library(GFLAGS_LIB gflags)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC ")

if(NOT CAFFE2_LIB)
  message(FATAL_ERROR "Caffe2 lib not found")
endif()

set(ALL_LIBRARIES)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BUILD_DIR}")

macro(link_whole_archive lib)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    list(APPEND ALL_LIBRARIES -Wl,-force_load,$<TARGET_FILE_NAME:${lib}> ${lib})
  elseif(MSVC)
    list(APPEND ALL_LIBRARIES -WHOLEARCHIVE:$<TARGET_FILE_NAME:${lib}>)
  else()
    list(APPEND ALL_LIBRARIES -Wl,--whole-archive ${lib} -Wl,--no-whole-archive)
  endif()
endmacro()

# link_whole_archive(${CAFFE2_LIB})

list(APPEND ALL_LIBRARIES ${CAFFE2_LIB})

list(APPEND ALL_LIBRARIES ${PROTOBUF_LIBRARY})

list(APPEND ALL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

list(APPEND ALL_LIBRARIES ${GLOG_LIB} ${GFLAGS_LIB})

if(OpenCV_LIBS)
  list(APPEND ALL_LIBRARIES ${OpenCV_LIBS})
  add_definitions(-DWITH_OPENCV)
endif()

add_executable(test_caffe2_and_opencv_cpp test_caffe2_and_opencv_cpp.cpp)
target_link_libraries(test_caffe2_and_opencv_cpp ${ALL_LIBRARIES})