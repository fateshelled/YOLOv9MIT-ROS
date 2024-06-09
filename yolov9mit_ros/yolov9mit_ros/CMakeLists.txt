cmake_minimum_required(VERSION 3.8)
project(yolov9mit_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

if(NOT yolov9mit_FOUND)
  message(WARNING "skipping yolov9mit_ros, no yolov9mit found")
  return()
endif()

ament_auto_add_library(yolov9mit_ros SHARED
  src/yolov9mit_ros.cpp
)
rclcpp_components_register_node(
  yolov9mit_ros
  PLUGIN "yolov9mit_ros::YOLOV9MIT_Node"
  EXECUTABLE yolov9mit_ros_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  # INSTALL_TO_SHARE
  # launch
)
