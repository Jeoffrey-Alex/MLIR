add_mlir_conversion_library(MLIRAlexToArith
    AlexToArith.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Alex

    LINK_LIBS PUBLIC
    MLIRAlex
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRIR
    MLIRPass
    MLIRRewrite
    MLIRTransformUtils
)