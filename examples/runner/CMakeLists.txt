# ExecuTorch's executor_runner (portable kernels) with RknnBackend linked in.
cmake_minimum_required(VERSION 3.24)
project(executorch_rknn_runner C CXX)

# executorch_core does not export its C++17 requirement, and GCC 8 defaults to C++14.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)
add_subdirectory(${ROOT}/third_party/executorch executorch)
add_subdirectory(${ROOT}/backend/runtime rknn_backend)

# The backend registers itself from a static initialiser nothing references,
# so the archive has to be linked whole.
target_link_libraries(
  executor_runner PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,executorch_rknn_backend>
)
