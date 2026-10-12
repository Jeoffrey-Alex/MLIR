// RUN: alex-opt %s | FileCheck %s

// CHECK: module
// CHECK: func.func @arange_test9() -> tensor<?xf32> attributes {llvm.emit_c_interface}
// CHECK: %[[END:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[RESULT:.*]] = "alex.arange"(%[[END]]) <{start = 5.000000e+00 : f32, step = -1.000000e+00 : f32}> : (f32) -> tensor<?xf32>
// CHECK: return %[[RESULT]] : tensor<?xf32>
module {
  func.func @arange_test9() -> tensor<?xf32> attributes { llvm.emit_c_interface } {
    %end = arith.constant 0.0 : f32

    %result = "alex.arange"(%end) {
      start = 5.0 : f32,
      step = -1.0 : f32
    } : (f32) -> tensor<?xf32>

    return %result : tensor<?xf32>
  }
}