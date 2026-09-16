add_mlir_public_c_api_library(MLIRAlexCAPI
    Dialects.cpp
    Passes.cpp

    LINK_LIBS PUBLIC
    MLIRAlex
    MLIRAlexToArith
)

