include "mlir/IR/OpBase.td"
include "Alex/AlexDialect.td"


class Alex_Op<string mnemonic, list<Trait> traits = []> : Op<Alex_Dialect, mnemonic, traits>;

def AddOp : Alex_Op<"add"> {
    let summary = "Adds two tensors with an optional scaling factor";
    let description = [{
    Adds two input tensors element-wise. The result is computed as:
        output = input1 + alpha * input2
    where `alpha` is an optional floating-point attribute that defaults to 1.0.
    }];

    // alex.add takes two tensor operands and an optional alpha attribute.
    let arguments = (ins
        TensorOf<[F32]>:$input1,
        TensorOf<[F32]>:$input2,
        DefaultValuedAttr<F32Attr, "1.0">:$alpha
    );

    let results = (outs
        TensorOf<[F32]>:$output
    );

}