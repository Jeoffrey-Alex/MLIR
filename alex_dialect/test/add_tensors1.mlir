// RUN: alex-opt %s | FileCheck %s

module {
    func.func @Add1Tensor() -> tensor<2x2xf32> attributes { llvm.emit_c_interface } {
        %a = "alex.constant"() {value = dense<{{INPUT1}}> : tensor<2x2xf32>} : () -> tensor<2x2xf32>

        %b = "alex.constant"() {value = {{INPUT2}} : f32} : () -> f32

        %result = "alex.add"(%a, %b) : (tensor<2x2xf32>, f32) -> tensor<2x2xf32>

        return %result : tensor<2x2xf32>
    }
}

// CHECK: module
// CHECK: func.func @Add1Tensor()
// CHECK: %[[A:.*]] = "alex.constant"
// CHECK: %[[B:.*]] = "alex.constant"
// CHECK: %[[RESULT:.*]] = "alex.add"(%[[A]], %[[B]]) : (tensor<2x2xf32>, f32) -> tensor<2x2xf32>
// CHECK: return %[[RESULT]] : tensor<2x2xf32>