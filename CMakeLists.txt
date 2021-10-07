project(mschol)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE RELEASE)
# list(APPEND CMAKE_CXX_FLAGS "-fopenmp -std=c++14 -fpermissive")
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

# lapack
find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    message("-- Lapack @ ${LAPACK_LIBRARIES}")
endif()

find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("-- BLAS @ ${BLAS_LIBRARIES}")
endif()

# header-only lib
include_directories(external/spdlog/include)
include_directories(external/egl/include)

add_executable(explicit explicit.cpp)
target_link_libraries(explicit ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})


add_executable(laplace laplace.cpp)
target_link_libraries(laplace ${OpenCV_LIBRARIES} Eigen3::Eigen)

add_executable(implicit implicit.cpp)
target_link_libraries(implicit ${OpenCV_LIBRARIES} Eigen3::Eigen ${Boost_LIBRARIES})

add_executable(smooth2d smooth2d.cpp)
target_link_libraries(smooth2d ${OpenCV_LIBRARIES} Eigen3::Eigen ${Boost_LIBRARIES})

add_executable(smooth3d smooth3d.cpp ichol.cpp)
target_link_libraries(smooth3d ${OpenCV_LIBRARIES} Eigen3::Eigen ${Boost_LIBRARIES})

add_executable(test_lapack test_lapack.cpp)
target_link_libraries(test_lapack ${LAPACK_LIBRARIES} Eigen3::Eigen ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

add_executable(test_blas test_blas.cpp)
target_link_libraries(test_blas ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
