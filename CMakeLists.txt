project(mschol)
cmake_minimum_required(VERSION 2.8)
# http://c.biancheng.net/gcc/

set(CMAKE_BUILD_TYPE RELEASE)
# Eigen
find_package(Eigen3)

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message("-- OpenCV @ ${OpenCV_INCLUDE_DIRS}")
  message("-- OpenCV @ ${OpenCV_LIBRARIES}")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# BOOST
find_package(Boost COMPONENTS filesystem REQUIRED)
if(Boost_FOUND)
  message("-- Boost @ ${Boost_INCLUDE_DIRS}")
  message("-- Boost @ ${Boost_LIBRARIES}")
  include_directories(${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)


add_executable(explicit explicit.cpp)
target_link_libraries(explicit ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})


add_executable(laplace laplace.cpp)
target_link_libraries(laplace ${OpenCV_LIBRARIES} Eigen3::Eigen)

add_executable(implicit implicit.cpp)
target_link_libraries(implicit ${OpenCV_LIBRARIES} Eigen3::Eigen ${Boost_LIBRARIES})

