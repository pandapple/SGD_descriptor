cmake_minimum_required(VERSION 3.0.2)
project(sgd_desc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf_conversions
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(yaml-cpp REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(utils_lib src/utils.cpp)

# Precision-Recall evaluation on SemanticKITTI (place recognition / reloc)
add_executable(s_evaluate_matches_triangular s_evaluate_matches_triangular.cpp src/sgd.cpp)
target_link_libraries(s_evaluate_matches_triangular
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  yaml-cpp
  tbb
  utils_lib)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
  target_link_libraries(s_evaluate_matches_triangular stdc++fs)
endif()

# Sequential loop-closure detection on SemanticKITTI
add_executable(loop_closure_detection loop_closure_detection.cpp src/sgd.cpp)
target_link_libraries(loop_closure_detection
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  yaml-cpp
  tbb
  utils_lib)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
  target_link_libraries(loop_closure_detection stdc++fs)
endif()
