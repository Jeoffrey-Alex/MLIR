// RUN: alex-opt %s -alex-decompose-addcmul | FileCheck %s --check-prefix=DECOMP
// RUN: alex-opt %s -alex-decompose-addcmul -convert-alex-to-arith | FileCheck %s --check-prefix=LOWER

module {
  func.func @AddcmulBroadcast() -> tensor<2x2xf32> attributes { llvm.emit_c_interface } {
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

// DECOMP-LABEL: func.func @AddcmulBroadcast
// DECOMP: "alex.mul"({{.*}}) : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
// DECOMP: "alex.constant"() <{value = dense<2.000000e+00> : tensor<2x2xf32>}>
// DECOMP: "alex.mul"({{.*}}) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// DECOMP: "alex.add"({{.*}}) : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// DECOMP-NOT: alex.addcmul

// LOWER: #[[MAP0:.*]] = affine_map<(d0, d1) -> (0, d1)>
// LOWER: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// LOWER: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// LOWER-LABEL: func.func @AddcmulBroadcast
// LOWER: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// LOWER: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// LOWER: linalg.elementwise kind=#linalg.elementwise_kind<add>
// LOWER-NOT: alex.
