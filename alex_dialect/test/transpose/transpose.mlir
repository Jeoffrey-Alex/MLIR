// RUN: alex-opt --convert-alex-to-linalg %s | FileCheck %s

// CHECK-DAG: #[[SWAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>

module {
  // 2-D: 2x3 -> 3x2
  // CHECK-LABEL: func.func @transpose_f32
  func.func @transpose_f32() -> tensor<3x2xf32> attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<3x2xf32>
    // CHECK: linalg.generic
    // CHECK-SAME: indexing_maps = [#[[SWAP]], #[[ID]]]
    // CHECK-SAME: iterator_types = ["parallel", "parallel"]
    // CHECK-SAME: outs(%[[EMPTY]] : tensor<3x2xf32>)
    // CHECK: ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
    // CHECK-NEXT: linalg.yield %[[IN]] : f32
    // CHECK-NOT: alex.transpose
    %result = "alex.transpose"(%input) {dim0 = 0 : i32, dim1 = 1 : i32}
        : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %result : tensor<3x2xf32>
  }
}
