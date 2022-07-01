# Copyright 2022 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cmake_minimum_required(VERSION 3.1)
project(TorchOpt LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# include(FindCUDA/select_compute_arch)
# CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
# string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
# string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
# string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
# message("-- nvcc generates code for arch ${CUDA_ARCH_LIST}")
# SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
SET(CMAKE_CUDA_ARCHITECTURES 53;60;61;70;75;80;86)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -fPIC -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

function(system)
    set(options STRIP)
    set(oneValueArgs OUTPUT_VARIABLE ERROR_VARIABLE WORKING_DIRECTORY)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(SYSTEM
                          "${options}"
                          "${oneValueArgs}"
                          "${multiValueArgs}"
                          "${ARGN}")

    if (NOT DEFINED SYSTEM_WORKING_DIRECTORY)
        set(SYSTEM_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif()

    execute_process(
        COMMAND ${SYSTEM_COMMAND}
        OUTPUT_VARIABLE STDOUT
        ERROR_VARIABLE STDERR
        WORKING_DIRECTORY "${SYSTEM_WORKING_DIRECTORY}"
    )
    if("${SYSTEM_STRIP}")
        string(STRIP "${STDOUT}" STDOUT)
        string(STRIP "${STDERR}" STDERR)
    endif()

    set("${SYSTEM_OUTPUT_VARIABLE}" "${STDOUT}" PARENT_SCOPE)
    if (DEFINED SYSTEM_ERROR_VARIABLE)
        set("${SYSTEM_ERROR_VARIABLE}" "${STDERR}" PARENT_SCOPE)
    endif()
endfunction()

if (NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python3)
endif()

system(
    STRIP OUTPUT_VARIABLE PYTHON_EXECUTABLE
    COMMAND which "${PYTHON_EXECUTABLE}"
)

system(
    STRIP OUTPUT_VARIABLE PYTHON_VERSION
    COMMAND "${PYTHON_EXECUTABLE}" --version
)

message("-- Use Python version: ${PYTHON_VERSION}")
message("-- Use Python executable: \"${PYTHON_EXECUTABLE}\"")

if(NOT DEFINED PYTHON_INCLUDE_DIR)
    message("-- Auto detecting Python include directory...")
    system(
        STRIP OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('include'))"
    )
endif()

if("${PYTHON_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "-- Python include directory not found")
else()
    message("-- Detected Python include directory: \"${PYTHON_INCLUDE_DIR}\"")
    include_directories(${PYTHON_INCLUDE_DIR})
endif()

if(NOT DEFINED TORCH_INCLUDE_PATH)
    message("-- Auto detecting PyTorch include directory...")
    system(
        STRIP OUTPUT_VARIABLE TORCH_INCLUDE_PATH
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('torch.utils.cpp_extension', fromlist=[None]).include_paths()[0])"
    )
endif()

if("${TORCH_INCLUDE_PATH}" STREQUAL "")
    message(FATAL_ERROR "-- Torch include directory not found")
else()
    message("-- Detected Torch include directory: \"${TORCH_INCLUDE_PATH}\"")
    include_directories(${TORCH_INCLUDE_PATH})
endif()

if(NOT DEFINED TORCH_LIBRARY_PATH)
    message("-- Auto detecting PyTorch library directory...")
    system(
        STRIP OUTPUT_VARIABLE TORCH_LIBRARY_PATH
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('torch.utils.cpp_extension', fromlist=[None]).library_paths()[0])"
    )
endif()

if("${TORCH_LIBRARY_PATH}" STREQUAL "")
    message(FATAL_ERROR "-- Torch library directory not found")
else()
    message("-- Detected Torch library directory: \"${TORCH_LIBRARY_PATH}\"")
endif()

unset(TORCH_LIBRARIES)
foreach(VAR_PATH ${TORCH_LIBRARY_PATH})
    file(GLOB TORCH_LIBRARY "${VAR_PATH}/*.so")
    list(APPEND TORCH_LIBRARIES "${TORCH_LIBRARY}")
endforeach()
message("-- Detected Torch libraries: \"${TORCH_LIBRARIES}\"")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_subdirectory("third_party/pybind11")
include_directories(include)

add_subdirectory(src)
