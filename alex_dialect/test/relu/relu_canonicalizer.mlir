// RUN: alex-opt %s -canonicalize -split-input-file | FileCheck %s

// relu after transpose: relu moves up, transpose moves down.
// CHECK-LABEL: func.func @relu_of_transpose
// CHECK-SAME: (%[[X:.*]]: tensor<2x3xf32>)
// CHECK: %[[R:.*]] = "alex.relu"(%[[X]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK: %[[T:.*]] = "alex.transpose"(%[[R]])
// CHECK: return %[[T]]
func.func @relu_of_transpose(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %t = "alex.transpose"(%x) {dim0 = 0 : i32, dim1 = 1 : i32}
      : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %r = "alex.relu"(%t) : (tensor<3x2xf32>) -> tensor<3x2xf32>
  return %r : tensor<3x2xf32>
}