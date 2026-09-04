set(LLVM_TARGET_DEFINITIONS Alex/AlexDialect.td)

mlir_tablegen(Alex/AlexDialect.h.inc -gen-dialect-decls)
mlir_tablegen(Alex/AlexDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS Alex/AlexOps.td)

mlir_tablegen(Alex/AlexOps.h.inc -gen-op-decls)
mlir_tablegen(Alex/AlexOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(AlexIncGen)