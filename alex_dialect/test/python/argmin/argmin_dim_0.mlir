// Test case for dim = 0

// RUN: alex-opt --convert-alex-to-arith %s | FileCheck %s

module {
  func.func @argmin_test() -> tensor<3xi64> attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[
      [1, 2, 3],
      [4, 0, 6],
      [7, 8, 9]
    ]> : tensor<3x3xi32>

    %result = "alex.argmin"(%input) {
      dim = 0 : i32
    } : (tensor<3x3xi32>) -> tensor<3xi64>

    return %result : tensor<3xi64>
  }
}

// CHECK-LABEL: func.func @argmin_test() -> tensor<3xi64>

// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<3xi64>

// Loop over non-reduced dimension (dim = 1)
 // CHECK: scf.for

// First value along reduction dimension
// CHECK: tensor.extract

// Reduction loop over dim = 0
// CHECK: scf.for

// Read current element
// CHECK: tensor.extract

// Compare current value with minimum
// CHECK: arith.cmpi slt

// Keep minimum value
// CHECK: arith.select

// Keep index of minimum
// CHECK: arith.index_cast
// CHECK: arith.select

// Carry reduction values
// CHECK: scf.yield

// Insert final index into result
// CHECK: tensor.insert

// CHECK: return