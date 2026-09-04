add_executable(alex-opt
    alex-opt.cpp
)

target_link_libraries(alex-opt
    PRIVATE
    MLIRAlex
    MLIROptLib
    MLIRArithDialect
    MLIRFuncDialect
)