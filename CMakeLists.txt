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

cmake_minimum_required(VERSION 3.8)
project(torchopt LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -fPIC -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(CUDA)

if(CUDA_FOUND)
    message(STATUS "Found CUDA, enabling CUDA support.")
    enable_language(CUDA)

    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS All)
    list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
else()
    message(STATUS "CUDA not found, build for CPU-only.")
endif()

function(system)
    set(options STRIP)
    set(oneValueArgs OUTPUT_VARIABLE ERROR_VARIABLE WORKING_DIRECTORY)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(
        SYSTEM
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        "${ARGN}"
    )

    if(NOT DEFINED SYSTEM_WORKING_DIRECTORY)
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

    if(DEFINED SYSTEM_ERROR_VARIABLE)
        set("${SYSTEM_ERROR_VARIABLE}" "${STDERR}" PARENT_SCOPE)
    endif()
endfunction()

if(NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python3)
endif()

system(
    STRIP OUTPUT_VARIABLE PYTHON_EXECUTABLE
    COMMAND bash -c "type -P '${PYTHON_EXECUTABLE}'"
)

system(
    STRIP OUTPUT_VARIABLE PYTHON_VERSION
    COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('platform').python_version())"
)

message(STATUS "Use Python version: ${PYTHON_VERSION}")
message(STATUS "Use Python executable: \"${PYTHON_EXECUTABLE}\"")

if(NOT DEFINED PYTHON_INCLUDE_DIR)
    message(STATUS "Auto detecting Python include directory...")
    system(
        STRIP OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('include'))"
    )
endif()

if("${PYTHON_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "Python include directory not found")
else()
    message(STATUS "Detected Python include directory: \"${PYTHON_INCLUDE_DIR}\"")
    include_directories(${PYTHON_INCLUDE_DIR})
endif()

set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}")

if(NOT DEFINED PYBIND11_CMAKE_DIR)
    message(STATUS "Auto detecting pybind11 CMake directory...")
    system(
        STRIP OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --cmakedir
    )
endif()

if("${PYBIND11_CMAKE_DIR}" STREQUAL "")
    message(FATAL_ERROR "Pybind11 CMake directory not found")
else()
    message(STATUS "Detected Pybind11 CMake directory: \"${PYBIND11_CMAKE_DIR}\"")
    find_package(pybind11 CONFIG PATHS "${PYBIND11_CMAKE_DIR}")
endif()

if(NOT DEFINED TORCH_INCLUDE_PATH)
    message(STATUS "Auto detecting PyTorch include directory...")
    system(
        STRIP OUTPUT_VARIABLE TORCH_INCLUDE_PATH
        COMMAND "${PYTHON_EXECUTABLE}" -c "print('\\\;'.join(__import__('torch.utils.cpp_extension', fromlist=[None]).include_paths()))"
    )
endif()

if("${TORCH_INCLUDE_PATH}" STREQUAL "")
    message(FATAL_ERROR "Torch include directory not found")
else()
    message(STATUS "Detected Torch include directory: \"${TORCH_INCLUDE_PATH}\"")
    include_directories(${TORCH_INCLUDE_PATH})
endif()

if(NOT DEFINED TORCH_LIBRARY_PATH)
    message(STATUS "Auto detecting PyTorch library directory...")
    system(
        STRIP OUTPUT_VARIABLE TORCH_LIBRARY_PATH
        COMMAND "${PYTHON_EXECUTABLE}" -c "print('\\\;'.join(__import__('torch.utils.cpp_extension', fromlist=[None]).library_paths()))"
    )
endif()

if("${TORCH_LIBRARY_PATH}" STREQUAL "")
    message(FATAL_ERROR "Torch library directory not found")
else()
    message(STATUS "Detected Torch library directory: \"${TORCH_LIBRARY_PATH}\"")
endif()

unset(TORCH_LIBRARIES)

foreach(VAR_PATH ${TORCH_LIBRARY_PATH})
    list(APPEND TORCH_LIBRARIES "${VAR_PATH}/libtorch_python.so")
endforeach()

message(STATUS "Detected Torch libraries: \"${TORCH_LIBRARIES}\"")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

include_directories(${CMAKE_SOURCE_DIR})
add_subdirectory(src)
