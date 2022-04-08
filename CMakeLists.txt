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
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -fPIC -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}  -O3")

if(NOT DEFINED PYTHON_INCLUDE_DIR)
    message("-- Auto detecting Python include directory...")
    execute_process (
        COMMAND python3 -c "import sys; import os; path = sys.path[2]; splited_path = path.split('/'); splited_path[-2] = 'include'; print(os.path.join('/', *splited_path))"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
    string(STRIP ${PYTHON_INCLUDE_DIR} PYTHON_INCLUDE_DIR)
endif()

if("${PYTHON_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "-- Python include directory not found")
else()
    message("-- Detected Python include directory: ${PYTHON_INCLUDE_DIR}")
    include_directories(${PYTHON_INCLUDE_DIR})
endif()

if(NOT DEFINED TORCH_INCLUDE_PATH)
    message("-- Auto detecting PyTorch include directory...")
    execute_process (
        COMMAND python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0], end='')"
        OUTPUT_VARIABLE TORCH_INCLUDE_PATH)
    string(STRIP ${TORCH_INCLUDE_PATH} TORCH_INCLUDE_PATH)
endif()

if("${TORCH_INCLUDE_PATH}" STREQUAL "")
    message(FATAL_ERROR "-- Torch include directory not found")
else()
    message("-- Detected Torch include directory: ${TORCH_INCLUDE_PATH}")
    include_directories(${TORCH_INCLUDE_PATH})
endif()


if(NOT DEFINED TORCH_LIBRARY_PATH)
    message("-- Auto detecting PyTorch library directory...")
    execute_process (
        COMMAND python3 -c "from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0], end='')"
        OUTPUT_VARIABLE TORCH_LIBRARY_PATH)
    string(STRIP ${TORCH_LIBRARY_PATH} TORCH_LIBRARY_PATH)
endif()

if("${TORCH_LIBRARY_PATH}" STREQUAL "")
    message(FATAL_ERROR "-- Torch library directory not found")
else()
    message("-- Detected Torch library directory: ${TORCH_LIBRARY_PATH}")
endif()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_subdirectory("third_party/pybind11")
include_directories(include)

foreach(TMP_PATH ${TORCH_LIBRARY_PATH})
    file(GLOB TORCH_LIBRARY ${TMP_PATH}/*.so)
    set(TORCH_LIBRARIES "${TORCH_LIBRARIES};${TORCH_LIBRARY};")
endforeach()

add_subdirectory(src)
