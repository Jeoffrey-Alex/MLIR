add_mlir_library(MLIRAlexTransforms
    DecomposeAddcmul.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Alex

    LINK_LIBS PUBLIC
    MLIRAlex
    MLIRDialect
    MLIRIR
    MLIRPass
    MLIRRewrite
    MLIRTransformUtils
)
