# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.21)
project(merlin-hkvs LANGUAGES CXX HIP)
find_package(HIP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Include the Hipify module early
include(${PROJECT_SOURCE_DIR}/cmake/modules/Hipify.cmake)

# Run hipification as an initial step
# This will hipify all CUDA files in the project and generate corresponding HIP files
set(excluded_patterns "tests/googletest;build;deps")  # Exclude tests/googletest third party dependency
hipify(
  ${CMAKE_CURRENT_SOURCE_DIR}  # Root directory to hipify
  excluded_patterns            # Variable containing excluded patterns
  generated_cc_files          # Output variable for generated .cc/.h files
  generated_cu_files          # Output variable for generated .cu files
)

option(CLANGFORMAT "Clangformat code files before compiling" OFF)
if(CLANGFORMAT)
  include(ClangFormat)
  # Update file patterns to include both original and hipified files
  file(GLOB_RECURSE clangformat_includes
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${PROJECT_SOURCE_DIR}/include/*.hpp
    ${PROJECT_SOURCE_DIR}/include/*.cuh
  )
  file(GLOB clangformat_tests
    ${PROJECT_SOURCE_DIR}/tests/*.c
    ${PROJECT_SOURCE_DIR}/tests/*.h
    ${PROJECT_SOURCE_DIR}/tests/*.cpp
    ${PROJECT_SOURCE_DIR}/tests/*.hpp
    ${PROJECT_SOURCE_DIR}/tests/*.cu
    ${PROJECT_SOURCE_DIR}/tests/*.cuh
  )
  set(clangformat_files ${clangformat_includes} ${clangformat_tests})
  clangformat_setup("${clangformat_files}")
endif()

# Default to release build.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Setting default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

# HIP doesn't have an equivalent to CUDA_SEPARABLE_COMPILATION
# This is handled differently in HIP compilation model

# Select target GPU architecture.
# ROCm uses gfx notation instead of sm
foreach(gpu_arch ${gfx})
  list(APPEND gpu_arch_list ${gpu_arch})
  message(STATUS "Assign GPU architecture (gfx=${gpu_arch})")
endforeach()

list(LENGTH gpu_arch_list gpu_arch_list_length)
if(gpu_arch_list_length EQUAL 0)
# Default to MI100 architecture (you may want to change this based on your target GPU)
list(APPEND gpu_arch_list "gfx950")
message(STATUS "Assign default GPU architecture gfx950")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
add_compile_definitions(HIP_ERROR_CHECK)
  # HIP equivalent for debug info
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g")
endif()

# Set HIP architectures
set(CMAKE_HIP_ARCHITECTURES ${gpu_arch_list})

# Alternative: manually set flags (older CMake versions)
# foreach(gpu_arch ${gpu_arch_list})
#   set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=${gpu_arch}")
# endforeach()

message("HIP_COMPILER=${HIP_COMPILER}")
message("CMAKE_HIP_FLAGS=${CMAKE_HIP_FLAGS}")
message("CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/tests/googletest/googletest/include
  ${HIP_INCLUDE_DIRS}
)

ADD_SUBDIRECTORY(tests/googletest)

link_directories(
  ${HIP_LIBRARY_DIRS}
)

# Update the file glob to use hipified files
file(GLOB_RECURSE merlin_hkvs_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  *.cpp 
  *.cu
  *rocm*.cc  # Include hipified files
  *rocm*.cu  # Include hipified CUDA files
)

# TODO:
# add_library(hierarchical_kv STATIC ${hierarchical_kv_src})
# target_compile_features(hierarchical_kv PUBLIC cxx_std_17)
# target_link_libraries(hierarchical_kv PUBLIC ...)

# TODO add libhipcxx dependency
find_package(libhipcxx)

add_compile_definitions(
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP
    __HIP_PLATFORM_AMD__
)


add_executable(merlin_hashtable_benchmark benchmark/merlin_hashtable_benchmark.cc.cu)
target_compile_features(merlin_hashtable_benchmark PUBLIC cxx_std_17)
set_target_properties(merlin_hashtable_benchmark PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
target_link_libraries(merlin_hashtable_benchmark hip::device libhipcxx::libhipcxx)

# add_executable(find_with_missed_keys_benchmark benchmark/find_with_missed_keys_benchmark.cc.cu)
# target_compile_features(find_with_missed_keys_benchmark PUBLIC cxx_std_17)
# set_target_properties(find_with_missed_keys_benchmark PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(find_with_missed_keys_benchmark hip::device)

# add_executable(merlin_hashtable_test tests/merlin_hashtable_test.cc.cu)
# target_compile_features(merlin_hashtable_test PUBLIC cxx_std_17)
# set_target_properties(merlin_hashtable_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(merlin_hashtable_test gtest_main hip::device)

# add_executable(find_or_insert_test tests/find_or_insert_test.cc.cu)
# target_compile_features(find_or_insert_test PUBLIC cxx_std_17)
# set_target_properties(find_or_insert_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(find_or_insert_test gtest_main hip::device)

# add_executable(merlin_memory_pool_test tests/memory_pool_test.cc.cu)
# target_compile_features(merlin_memory_pool_test PUBLIC cxx_std_17)
# set_target_properties(merlin_memory_pool_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(merlin_memory_pool_test gtest_main hip::device)

# set(CMAKE_BUILD_TYPE "Debug")
# add_executable(save_and_load_test tests/save_and_load_test.cc.cu)
# target_compile_features(save_and_load_test PUBLIC cxx_std_17)
# set_target_properties(save_and_load_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(save_and_load_test gtest_main hip::device)

# add_executable(insert_and_evict_test tests/insert_and_evict_test.cc.cu)
# target_compile_features(insert_and_evict_test PUBLIC cxx_std_17)
# set_target_properties(insert_and_evict_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(insert_and_evict_test gtest_main hip::device)

# add_executable(dynamic_max_capacity_test tests/dynamic_max_capacity_test.cc.cu)
# target_compile_features(dynamic_max_capacity_test PUBLIC cxx_std_17)
# set_target_properties(dynamic_max_capacity_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(dynamic_max_capacity_test gtest_main hip::device)

# add_executable(group_lock_test tests/group_lock_test.cc.cu)
# target_compile_features(group_lock_test PUBLIC cxx_std_17)
# set_target_properties(group_lock_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(group_lock_test gtest_main hip::device)

# add_executable(find_or_insert_ptr_test tests/find_or_insert_ptr_test.cc.cu)
# target_compile_features(find_or_insert_ptr_test PUBLIC cxx_std_17)
# set_target_properties(find_or_insert_ptr_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(find_or_insert_ptr_test gtest_main hip::device)

# add_executable(assign_score_test tests/assign_score_test.cc.cu)
# target_compile_features(assign_score_test PUBLIC cxx_std_17)
# set_target_properties(assign_score_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(assign_score_test gtest_main hip::device)

# add_executable(accum_or_assign_test tests/accum_or_assign_test.cc.cu)
# target_compile_features(accum_or_assign_test PUBLIC cxx_std_17)
# set_target_properties(accum_or_assign_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(accum_or_assign_test gtest_main hip::device)

# add_executable(assign_values_test tests/assign_values_test.cc.cu)
# target_compile_features(assign_values_test PUBLIC cxx_std_17)
# set_target_properties(assign_values_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(assign_values_test gtest_main hip::device)

# add_executable(find_with_missed_keys_test tests/find_with_missed_keys_test.cc.cu)
# target_compile_features(find_with_missed_keys_test PUBLIC cxx_std_17)
# set_target_properties(find_with_missed_keys_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(find_with_missed_keys_test gtest_main hip::device)

# add_executable(reserved_keys_test tests/reserved_keys_test.cc.cu)
# target_compile_features(reserved_keys_test PUBLIC cxx_std_17)
# set_target_properties(reserved_keys_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(reserved_keys_test gtest_main hip::device)

# add_executable(export_batch_if_test tests/export_batch_if_test.cc.cu)
# target_compile_features(export_batch_if_test PUBLIC cxx_std_17)
# set_target_properties(export_batch_if_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(export_batch_if_test hip::device)

# add_executable(find_or_insert_ptr_lock_test tests/find_or_insert_ptr_lock_test.cc.cu)
# target_compile_features(find_or_insert_ptr_lock_test PUBLIC cxx_std_17)
# set_target_properties(find_or_insert_ptr_lock_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(find_or_insert_ptr_lock_test gtest_main hip::device)

# add_executable(lock_unlock_test tests/lock_unlock_test.cc.cu)
# target_compile_features(lock_unlock_test PUBLIC cxx_std_17)
# set_target_properties(lock_unlock_test PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
# target_link_libraries(lock_unlock_test gtest_main hip::device)
