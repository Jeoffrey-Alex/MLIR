// RUN: alex-opt %s -canonicalize | FileCheck %s

// (-1, 0) on 2x3 is rewritten to the standard form (0, 1).
// CHECK-LABEL: func.func @normalize_dims
// CHECK: "alex.transpose"(%{{.*}}) <{dim0 = 0 : i32, dim1 = 1 : i32}>
// CHECK-SAME: (tensor<2x3xf32>) -> tensor<3x2xf32>
func.func @normalize_dims(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %a = "alex.transpose"(%x) {dim0 = -1 : i32, dim1 = 0 : i32}
      : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %a : tensor<3x2xf32>
}
