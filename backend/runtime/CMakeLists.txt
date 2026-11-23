# RknnBackend for ExecuTorch. Expects to be added from a project that has
# already added ExecuTorch (target executorch_core), and RKNPU2 set to the root
# of rknn-toolkit2's rknpu2 directory.
cmake_minimum_required(VERSION 3.24)

set(RKNPU2 "$ENV{RKNPU2}" CACHE PATH "rknn-toolkit2 rknpu2 directory")
if(NOT EXISTS "${RKNPU2}/runtime/Linux/librknn_api/include/rknn_api.h")
  message(FATAL_ERROR "RKNPU2 does not point at rknpu2: ${RKNPU2}")
endif()

add_library(rknnmrt SHARED IMPORTED)
set_target_properties(rknnmrt PROPERTIES
  IMPORTED_LOCATION "${RKNPU2}/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so"
  INTERFACE_INCLUDE_DIRECTORIES "${RKNPU2}/runtime/Linux/librknn_api/include")

add_library(executorch_rknn_backend STATIC RknnBackend.cpp)
target_compile_features(executorch_rknn_backend PRIVATE cxx_std_17)
target_link_libraries(executorch_rknn_backend PRIVATE executorch_core PUBLIC rknnmrt)
