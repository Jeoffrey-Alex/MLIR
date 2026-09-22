// RUN: alex-opt %s | FileCheck %s

// CHECK: module
// CHECK: func.func @arange_test1() -> tensor<?xi32> attributes {llvm.emit_c_interface}
// CHECK: %[[END:.*]] = arith.constant 5 : i32
// CHECK: %[[RESULT:.*]] = "alex.arange"(%[[END]]) <{start = 0 : i32, step = 1 : i32}> : (i32) -> tensor<?xi32>
// CHECK: return %[[RESULT]] : tensor<?xi32>
module {
  func.func @arange_test1() -> tensor<?xi32> attributes { llvm.emit_c_interface }{
    %end = arith.constant 5 : i32

    %result = "alex.arange"(%end) {
      start = 0 : i32,
      step = 1 : i32
    } : (i32) -> tensor<?xi32>

    return %result : tensor<?xi32>
  }
}