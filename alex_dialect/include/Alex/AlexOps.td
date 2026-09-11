include "mlir/IR/OpBase.td"
include "Alex/AlexDialect.td"

def IntOrFloatTensor : TensorOf<[AnyTypeOf<[AnyInteger, AnyFloat]>]>;
def Float3DTensor : TensorRankOf<[AnyFloat], [3]>;

class Alex_Op<string mnemonic, list<Trait> traits = []>
    : Op<Alex_Dialect, mnemonic, traits>;

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

def PractiseOp : Alex_Op<"practise"> {
    let summary = "operation with two tensor inputs, one optional, and a float array attribute";

    let description = [{
        This operation takes two tensor operands and one attribute:

        - `input1` is a required tensor of either integer or floating-point element type.
        - `input2` is an optional tensor with the same type constraint as `input1`.
        - `testAttr` is a required array of `f32` values.

        The result `output` is a 3DFloat Tensor
    }];

    let arguments = (ins
        IntOrFloatTensor:$input1,
        Optional<IntOrFloatTensor>:$input2,
        F32ArrayAttr:$testAttr
    );

    let results = (outs
        Float3DTensor:$output
    );
}
