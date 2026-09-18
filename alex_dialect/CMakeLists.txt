cmake_minimum_required(VERSION 3.20)

project(MLIRAlex LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

if(MLIR_ENABLE_BINDINGS_PYTHON)
    message(STATUS "Enabling Python API")

    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()

    if(NOT MLIR_PYTHON_PACKAGE_PREFIX)
        set(MLIR_PYTHON_PACKAGE_PREFIX "mlir_alex" CACHE STRING "" FORCE)
    endif()

    if(NOT MLIR_BINDINGS_PYTHON_NB_DOMAIN)
        set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "mlir_alex" CACHE STRING "" FORCE)
    endif()

    if(NOT MLIR_BINDINGS_PYTHON_INSTALL_PREFIX)
        set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX
            "python_packages/alex/${MLIR_PYTHON_PACKAGE_PREFIX}"
            CACHE STRING "" FORCE)
    endif()

    add_subdirectory(python)
endif()

add_subdirectory(tools)
