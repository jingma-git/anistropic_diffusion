project(mschol)
cmake_minimum_required(VERSION 2.8)
# http://c.biancheng.net/gcc/

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message("-- OpenCV @ ${OpenCV_INCLUDE_DIRS}")
  message("-- OpenCV @ ${OpenCV_LIBRARIES}")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)


add_executable(explicit explicit.cpp)
target_link_libraries(explicit ${OpenCV_LIBRARIES})
