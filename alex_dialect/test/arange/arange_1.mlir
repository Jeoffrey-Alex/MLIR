// RUN: alex-opt %s | FileCheck %s

// CHECK: module
// CHECK: func.func @arange_test1() -> tensor<?xf32> attributes {llvm.emit_c_interface}
// CHECK: %[[END:.*]] = arith.constant 2.500000e+00 : f32
// CHECK: %[[RESULT:.*]] = "alex.arange"(%[[END]]) <{start = 1.100000e+00 : f32, step = 1 : i32}> : (f32) -> tensor<?xf32>
// CHECK: return %[[RESULT]] : tensor<?xf32>

module {
  func.func @arange_test1() -> tensor<?xf32> attributes { llvm.emit_c_interface } {
    %end = arith.constant 2.5 : f32

    %result = "alex.arange"(%end) {
      start = 1.1 : f32,
      step = 1 : i32
    } : (f32) -> tensor<?xf32>

    return %result : tensor<?xf32>
  }
}