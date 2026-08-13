cmake_minimum_required(VERSION 3.22)
project(manipulator_altro_benchmark LANGUAGES CXX)

include(FetchContent)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)

set(ALTRO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ALTRO_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(ALTRO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ALTRO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ALTRO_ENABLE_ARDUINO_LIB_INSTALL OFF CACHE BOOL "" FORCE)
set(ALTRO_ENABLE_RUNTIME_EXCEPTIONS ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  altro
  GIT_REPOSITORY https://github.com/bjack205/altro.git
  GIT_TAG 9819e66a6d20daf39e23bd9b45444b7f5aeb3322
  PATCH_COMMAND
    bash -c
    "git apply --check --ignore-whitespace '${CMAKE_CURRENT_SOURCE_DIR}/altro-cmake-3.22.patch' && git apply --ignore-whitespace '${CMAKE_CURRENT_SOURCE_DIR}/altro-cmake-3.22.patch' || git apply --reverse --check '${CMAKE_CURRENT_SOURCE_DIR}/altro-cmake-3.22.patch'"
)
FetchContent_MakeAvailable(altro)

set(MANIPULATOR_PACKAGE
    ${CMAKE_CURRENT_SOURCE_DIR}/../manipulator_workspace_collision_package)

add_executable(
  manipulator_altro_random_pose_benchmark
  manipulator_altro_random_pose_benchmark.cpp
)
target_compile_features(manipulator_altro_random_pose_benchmark PRIVATE cxx_std_17)
target_compile_options(manipulator_altro_random_pose_benchmark PRIVATE -O3)
target_include_directories(
  manipulator_altro_random_pose_benchmark PRIVATE
  ${MANIPULATOR_PACKAGE}/include
  ${MANIPULATOR_PACKAGE}/examples
  ${MANIPULATOR_PACKAGE}/examples/random_pose_benchmark
)
target_link_libraries(
  manipulator_altro_random_pose_benchmark PRIVATE
  altro::altro
  altrocpp::altro
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  PkgConfig::ZSTD
)
