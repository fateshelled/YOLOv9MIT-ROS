cmake_minimum_required(VERSION 3.8)
project(yolov9mit)

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

option(YOLOV9_MIT_USE_TENSORRT    "Use TensorRT"  ON)

set(ENABLE_TENSORRT OFF)

if(YOLOV9_MIT_USE_TENSORRT)
  find_package(CUDA REQUIRED)
  find_library(NVINFER  NAMES nvinfer REQUIRED)
  find_library(NVINFERPLUGIN NAMES nvinfer_plugin  REQUIRED)
  find_library(NVPARSERS NAMES nvparsers REQUIRED)
  find_library(NVONNXPARSER NAMES nvonnxparser REQUIRED)

  set(ENABLE_TENSORRT ON)
  set(TARGET_SRC src/yolov9mit_tensorrt.cpp)
  set(TARGET_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
  set(TARGET_DPENDENCIES CUDA)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/include/yolov9mit/config.h.in"
  "${PROJECT_SOURCE_DIR}/include/yolov9mit/config.h"
)

ament_auto_add_library(yolov9mit SHARED ${TARGET_SRC})
ament_target_dependencies(yolov9mit ${TARGET_DPENDENCIES})
ament_export_dependencies(${TARGET_DPENDENCIES})

if (YOLOV9_MIT_USE_TENSORRT)
  target_link_libraries(yolov9mit ${TARGET_LIBS})
endif()


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
