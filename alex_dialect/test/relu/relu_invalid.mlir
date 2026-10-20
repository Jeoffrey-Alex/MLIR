// RUN: alex-opt %s -split-input-file -verify-diagnostics

// Wrong shape: relu keeps the shape, so 2x3 can't become 3x2.
module
{
    func.func @wrong_shape(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
    // expected-error @+1 {{expected result type 'tensor<2x3xf32>', got 'tensor<3x2xf32>'}}
    %0 = "alex.relu"(%x) : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
    }
}
