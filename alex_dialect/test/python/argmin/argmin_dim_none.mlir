//Test case for dim = none

// RUN: alex-opt --convert-alex-to-arith %s | FileCheck %s

module {
  func.func @argmin_test() -> tensor<i64> attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[
      [1, 2, 3],
      [4, 0, 6],
      [7, 8, 9]
    ]> : tensor<3x3xi32>

    %result = "alex.argmin"(%input) {
    } : (tensor<3x3xi32>) -> tensor<i64>

    return %result : tensor<i64>
  }
}

// CHECK-LABEL: func.func @argmin_test() -> tensor<i64>

// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<i64>

// CHECK: %[[FIRST:.*]] = tensor.extract %{{.*}}[{{.*}}, {{.*}}] : tensor<3x3xi32>

// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: arith.index_cast
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK: arith.select
// CHECK: scf.yield

// CHECK: tensor.insert
// CHECK: return