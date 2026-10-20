// RUN: alex-opt %s -canonicalize | FileCheck %s

// relu twice: the outer one folds away, one relu remains.
// CHECK-LABEL: func.func @double_relu
// CHECK-SAME: (%[[X:.*]]: tensor<2x3xf32>)
// CHECK: %[[R:.*]] = alex.relu %[[X]]
// CHECK-NOT: alex.relu
// CHECK: return %[[R]]
func.func @double_relu(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %a = "alex.relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %b = "alex.relu"(%a) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %b : tensor<2x3xf32>
}