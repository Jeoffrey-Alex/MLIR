include "mlir/IR/OpBase.td"
include "Alex/AlexDialect.td"


class Alex_Op<string mnemonic, list<Trait> traits = []> : Op<Alex_Dialect, mnemonic, traits>;

def AddOp : Alex_Op<"add"> {
    let summary = "Add two integers";

    let arguments = (ins
        I32:$lhs,
        I32:$rhs
    );

    let results = (outs
        I32:$result
    );

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
