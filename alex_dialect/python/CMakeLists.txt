include(AddMLIRPython)

add_compile_definitions(
    "MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}."
)

declare_mlir_python_sources(AlexPythonSources)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT AlexPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_alex"
    TD_FILE dialects/AlexOps.td
    SOURCES
        dialects/alex_nanobind.py
    DIALECT_NAME alex
)

declare_mlir_python_extension(AlexPythonSources.NanobindExtension
    MODULE_NAME _alexDialectsNanobind
    ADD_TO_PARENT AlexPythonSources
    SOURCES 
        AlexExtensionNanobind.cpp

    EMBED_CAPI_LINK_LIBS
        MLIRCAPIIR
        MLIRCAPIArith
        MLIRCAPIControlFlow
        MLIRCAPITransforms
        MLIRCAPIFunc
        MLIRCAPIMath
        MLIRCAPIMemRef
        MLIRCAPISCF
        MLIRCAPILinalg
        MLIRCAPITensor
        MLIRCAPIBufferization
        MLIRCAPILLVM
        MLIRCAPIExecutionEngine
        MLIRAlexCAPI
)

add_mlir_python_common_capi_library(AlexPythonCAPI
    INSTALL_COMPONENT AlexPythonModules
    INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
    OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"    RELATIVE_INSTALL_ROOT "../../../.."
    DECLARED_SOURCES
        AlexPythonSources
        MLIRPythonSources.Core
        MLIRPythonSources.Dialects.builtin
        MLIRPythonSources.ExecutionEngine
)

add_mlir_python_modules(AlexPythonModules
    ROOT_PREFIX
        "${CMAKE_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
    INSTALL_PREFIX
        "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
    DECLARED_SOURCES
        AlexPythonSources
        MLIRPythonSources.Core
        MLIRPythonSources.Dialects.builtin
        MLIRPythonSources.ExecutionEngine
    COMMON_CAPI_LINK_LIBS
        AlexPythonCAPI
)
