// RUN: alex-opt %s -convert-alex-to-arith | FileCheck %s

module {
  func.func @Addcmul() -> tensor<2x2xf32> attributes { llvm.emit_c_interface } {
    %input = "alex.constant"() {
      value = dense<[[1.0, 2.0]]> : tensor<1x2xf32>
    } : () -> tensor<1x2xf32>

    %tensor1 = "alex.constant"() {
      value = dense<[[2.0, 3.0]]> : tensor<1x2xf32>
    } : () -> tensor<1x2xf32>

    %tensor2 = "alex.constant"() {
      value = dense<[[5.0], [6.0]]> : tensor<2x1xf32>
    } : () -> tensor<2x1xf32>

    %result = "alex.addcmul"(%input, %tensor1, %tensor2) {
      value = 2.0 : f32
    } : (
      tensor<1x2xf32>,
      tensor<1x2xf32>,
      tensor<2x1xf32>
    ) -> tensor<2x2xf32>

    return %result : tensor<2x2xf32>
  }
}
// CHECK: #map = affine_map<(d0, d1) -> (0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #map2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @Addcmul
// CHECK: linalg.fill
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-NOT: alex.addcmul