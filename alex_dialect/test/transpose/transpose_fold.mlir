// RUN: alex-opt %s -canonicalize | FileCheck %s

// Transpose twice with the same dims: 2x3 -> 3x2 -> 2x3.
// Both transposes disappear and the function returns its input.
// CHECK-LABEL: func.func @double_transpose
// CHECK-SAME: (%[[X:.*]]: tensor<2x3xf32>)
// CHECK-NOT: alex.transpose
// CHECK: return %[[X]]
func.func @double_transpose(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %a = "alex.transpose"(%x) {dim0 = 0 : i32, dim1 = 1 : i32}
      : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %b = "alex.transpose"(%a) {dim0 = 0 : i32, dim1 = 1 : i32}
      : (tensor<3x2xf32>) -> tensor<2x3xf32>
  return %b : tensor<2x3xf32>
}
