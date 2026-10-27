#ifndef ALEX_DIALECT_TD
#define ALEX_DIALECT_TD

include "mlir/IR/DialectBase.td"

def Alex_Dialect : Dialect {
    let name = "alex";
    let cppNamespace = "::alex";
    let useDefaultTypePrinterParser = 1;
}

#endif
