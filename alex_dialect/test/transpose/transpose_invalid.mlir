// RUN: alex-opt %s -split-input-file -verify-diagnostics

// Wrong result type: 2x3 transposed is 3x2, not 9x9.
func.func @wrong_result_2d(%x: tensor<2x3xf32>) -> tensor<9x9xf32> {
  // expected-error @+1 {{expected result type 'tensor<3x2xf32>', got 'tensor<9x9xf32>'}}
  %0 = "alex.transpose"(%x) {dim0 = 0 : i32, dim1 = 1 : i32}
      : (tensor<2x3xf32>) -> tensor<9x9xf32>
  return %0 : tensor<9x9xf32>
}

// -----

// Your bad.mlir: swapping dims 0 and 2 of 2x3x4 gives 4x3x2.
func.func @wrong_result_3d(%x: tensor<2x3x4xf32>) -> tensor<3x2x4xf32> {
  // expected-error @+1 {{expected result type 'tensor<4x3x2xf32>', got 'tensor<3x2x4xf32>'}}
  %0 = "alex.transpose"(%x) {dim0 = 0 : i32, dim1 = 2 : i32}
      : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
  return %0 : tensor<3x2x4xf32>
}

// -----

// A 2-D tensor only has dims 0 and 1, so dim 7 doesn't exist.
func.func @dim_out_of_range(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // expected-error @+1 {{dims (0, 7) out of range for rank 2}}
  %0 = "alex.transpose"(%x) {dim0 = 0 : i32, dim1 = 7 : i32}
      : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
