include "mlir/IR/OpBase.td"
include "Alex/AlexDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td" 


def IntOrFloatTensor : TensorOf<[AnyTypeOf<[AnyInteger, AnyFloat]>]>;
def Float3DTensor : TensorRankOf<[AnyFloat], [3]>;
def ScalarOrTensor : AnyTypeOf<[AnyFloat, AnyInteger, TensorOf<[AnyFloat, AnyInteger]>]>;
def AnyScalarType : AnyTypeOf<[AnyFloat,AnyInteger]>;

class Alex_Op<string mnemonic, list<Trait> traits = []> : Op<Alex_Dialect, mnemonic, traits>;

def PractiseOp : Alex_Op<"practise"> {
    let summary = "Operation with two tensor inputs and a float array attribute";

    let description = [{
        This operation takes the following operands and attribute:

        - `input1` is a required tensor with integer or floating-point elements.
        - `input2` is an optional tensor with integer or floating-point elements.
        - `testAttr` is a required array of `f32` values.

        The result `output` is a 3D floating-point tensor.
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

def AddOp : Alex_Op<"add"> {
    let summary = "Adds two scalar or tensor values";

    let description = [{
        This operation takes two operands:

        - `input1` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.
        - `input2` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.

        The result `output` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.
    }];

    let arguments = (ins
        ScalarOrTensor:$input1,
        ScalarOrTensor:$input2
    );

    let results = (outs
        ScalarOrTensor:$output
    );
}

def SubOp : Alex_Op<"sub"> {
    let summary = "Subtracts one scalar or tensor value from another";

    let description = [{
        This operation takes two operands:

        - `input1` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.
        - `input2` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.

        The result `output` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.
    }];

    let arguments = (ins
        ScalarOrTensor:$input1,
        ScalarOrTensor:$input2
    );

    let results = (outs
        ScalarOrTensor:$output
    );
}

def MulOp : Alex_Op<"mul"> {
    let summary = "Multiplies two scalar or tensor values";

    let description = [{
        This operation takes two operands:

        - `input1` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.
        - `input2` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.

        The result `output` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.
    }];

    let arguments = (ins
        ScalarOrTensor:$input1,
        ScalarOrTensor:$input2
    );

    let results = (outs
        ScalarOrTensor:$output
    );
}

def ConstOp : Alex_Op<"constant"> {
    let summary = "Creates a scalar or tensor constant";

    let description = [{
        This operation takes one attribute:

        - `value` is a required attribute of any attribute type.

        The result `result` can be a floating-point scalar, integer scalar, or tensor with floating-point or integer elements.
    }];

    let arguments = (ins
        AnyAttr:$value
    );

    let results = (outs
        ScalarOrTensor:$result
    );
}

def AddcmulOp : Alex_Op<"addcmul"> {
    let summary = "Adds the element-wise product of two tensors to an input tensor";

    let description = [{
        This operation takes three tensor operands and one attribute:

        - `input` is a tensor with integer or floating-point elements.
        - `tensor1` is a tensor with integer or floating-point elements.
        - `tensor2` is a tensor with integer or floating-point elements.
        - `value` is an `f32` attribute with a default value of `1.0`.

        The result `result` is a tensor with integer or floating-point elements.
    }];

    let arguments = (ins
        IntOrFloatTensor: $input,
        IntOrFloatTensor: $tensor1,
        IntOrFloatTensor: $tensor2,
        DefaultValuedAttr<F32Attr, "1.0">:$value 
    );

    let results = (outs
        IntOrFloatTensor:$result
    );
}

def addLinalgOp : Alex_Op<"addLinalg"> {
    let summary = [{
        Accumulates the sum of two tensors and divides by a scalar value.
    }];

    let description = [{
        This operation takes two operands:

        - `input1` is a tensor with integer or floating-point elements.
        - `input2` is a tensor with integer or floating-point elements.

        The result `result` is a tensor with integer or floating-point elements.
    }];

    let arguments = (ins
        IntOrFloatTensor:$input1,
        IntOrFloatTensor:$input2
    );

    let results = (outs
        IntOrFloatTensor:$result
    );
}

def addBCLinalgOp : Alex_Op<"addBCLinalg"> {
    let summary = [{
        Accumulates the sum of two tensors and divides by a scalar value.
    }];

    let description = [{
        This operation takes two operands:

        - `input1` is a tensor with integer or floating-point elements.
        - `input2` is a tensor with integer or floating-point elements.

        The result `result` is a tensor with integer or floating-point elements.
    }];

    let arguments = (ins
        IntOrFloatTensor:$input1,
        IntOrFloatTensor:$input2
    );

    let results = (outs
        IntOrFloatTensor:$result
    );
}

def reductionLinalgOp : Alex_Op<"reductionLinalg"> {
    let summary = [{
        Accumulates the sum of two tensors and divides by a scalar value.
    }];

    let description = [{
        This operation takes two operands:

        - `input1` is a tensor with integer or floating-point elements.
        - `input2` is a tensor with integer or floating-point elements.

        The result `result` is a tensor with integer or floating-point elements.
    }];

    let arguments = (ins
        IntOrFloatTensor:$input1
    );

    let results = (outs
        AnyTypeOf<[AnyInteger, AnyFloat]>:$result
    );
}

def argminOp: Alex_Op<"argmin"> {
    let summary = "Returns the indices of the minimum value of a tensor";

    let description = [{
        This operation takes a tensor operand and an integer attribute and bool attribute.
        
        - 'input' is a required tensor of any type and shape.
        - 'dim' is an optional attribute that specifies the dimension to reduce.
        - 'keepdim' is an attribute that specifies whether the reduced dimension is kept in the output with size 1, by default it is set to false.

        - 'result' is a tensor of integer values containing the indices of the minimum value.
        }];

    let arguments = (ins
        IntOrFloatTensor:$input,
        OptionalAttr<I32Attr>:$dim,
        DefaultValuedAttr<BoolAttr, "false">:$keepdim
    );

    let results = (outs
        I64Tensor:$result
    );
}

def reluOp: Alex_Op<"relu"> {
    let summary = "Applies the rectified linear unit function to a tensor";

    let description = [{
        This operation takes a tensor operand and applies the rectified linear unit (ReLU) function element-wise.

        - 'input' is a required tensor of any type and shape.
        - 'result' is a tensor of the same type and shape as the input, with all negative values replaced by zero.
    }];

    let arguments = (ins
        IntOrFloatTensor:$input
    );

    let results = (outs
        IntOrFloatTensor:$result
    );
}

def transposeOp: Alex_Op<"transpose",[Pure]>{
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "mlir::Value":$input, "int32_t":$dim0, "int32_t":$dim1)>
    ];

    let hasFolder = 1;

    let hasCanonicalizer = 1;

    let summary = "Swaps two dimensions of a tensor";

    let description = [{
        This operation takes a tensor operand and swaps two of its dimensions, like torch.transpose.

        - 'input' is a required ranked tensor of any rank >= 1.
        - 'dim0' and 'dim1' are the dimensions to swap. Negative values count from the end (-1 is the last dimension).
        - 'result' has the input shape with dimensions dim0 and dim1 exchanged, and the same element type.
    }];

    let arguments = (ins
        IntOrFloatTensor:$input,
        I32Attr:$dim0,
        I32Attr:$dim1
    );

    let results = (outs
        IntOrFloatTensor:$result    
    );
}