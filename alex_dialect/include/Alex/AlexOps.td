include "mlir/IR/OpBase.td"
include "Alex/AlexDialect.td"


class Alex_Op<string mnemonic, list<Trait> traits = []> : Op<Alex_Dialect, mnemonic, traits>;

def AddOp : Alex_Op<"add"> {

}


