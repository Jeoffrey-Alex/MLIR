//RUN: alex-opt %s --convert-alex-to-arith | FileCheck %s

//CHECK-LABEL: func.func @Scalar
//CHECK: arith.constant 1.000000e+00 : f32
//CHECK: arith.constant 2.000000e+00 : f32
//CHECK: arith.constant 3.000000e+00 : f32
//CHECK: arith.constant 4.000000e+00 : f32
//CHECK: arith.mulf
//CHECK: arith.addf
//CHECK: arith.subf
//CHECK: return

//CHECK-LABEL: func.func @Tensor
//CHECK: arith.constant dense<1.000000e+00> : tensor<2x2xf32>
//CHECK: arith.constant dense<2.000000e+00> : tensor<2x2xf32>
//CHECK: arith.constant dense<3.000000e+00> : tensor<2x2xf32>
//CHECK: arith.constant dense<4.000000e+00> : tensor<2x2xf32>
//CHECK: tensor.empty
//CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
//CHECK: tensor.empty
//CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
//CHECK: tensor.empty
//CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sub>
//CHECK: return

//CHECK-LABEL: func.func @ScalarMixed
//b*c
//CHECK: %[[MUL:.*]] = arith.mulf {{.*}},{{.*}} : f32
// a is i32, so convert it to f32
//CHECK: %[[CONV:.*]] = arith.sitofp{{.*}} : i32 to f32
// a + (b * c)
//CHECK: %[[ADD:.*]] = arith.addf %[[CONV]], %[[MUL]] : f32
// (a + (b * c)) - d
//CHECK: %[[SUB:.*]] = arith.subf %[[ADD]], {{.*}} : f32
// CHECK: return %[[SUB]] : f32


//CHECK-LABEL: func.func @TensorScalar
// b * c -> tensor
//CHECK: %[[FILL1:.*]] = linalg.fill
//CHECK: %[[MUL:.*]] = linalg.elementwise
// a + (b * c)
//CHECK: %[[ADD:.*]] = linalg.elementwise
// (a + (b * c)) - d
//CHECK: %[[FILL2:.*]] = linalg.fill
//CHECK: %[[SUB:.*]] =  linalg.elementwise
// CHECK: return


module {
    func.func @Scalar() -> f32 {
        %a = "alex.constant"(){value = 1.0 : f32}:() -> f32
        %b = "alex.constant"(){value = 2.0 : f32}:() -> f32
        %c = "alex.constant"(){value = 3.0 : f32}:() -> f32
        %d = "alex.constant"(){value = 4.0 : f32}:() -> f32

        %bc = "alex.mul"(%b,%c) : (f32, f32) -> f32
        %abc = "alex.add"(%a,%bc) : (f32,f32) -> f32
        %result = "alex.sub"(%abc,%d) : (f32, f32) -> f32

        return %result : f32
    }

    func.func@Tensor() -> tensor<2x2xf32> {
        %a = "alex.constant"(){value = dense<1.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
        %b = "alex.constant"(){value = dense<2.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
        %c = "alex.constant"(){value = dense<3.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
        %d = "alex.constant"(){value = dense<4.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>

        %bc = "alex.mul"(%b, %c) : (tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
        %abc = "alex.add"(%a, %bc) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %result = "alex.sub"(%abc, %d) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

        return %result : tensor<2x2xf32>
    }

    func.func @ScalarMixed() -> f32 {
        %a = "alex.constant"(){value = 1 : i32} : () -> i32
        %b = "alex.constant"(){value = 2.0 : f32} : () -> f32
        %c = "alex.constant"(){value = 3.0 : f32} : () -> f32
        %d = "alex.constant"(){value = 4.0 : f32} : () -> f32

        %bc = "alex.mul"(%b, %c) : (f32, f32) -> f32
        %abc = "alex.add"(%a, %bc) : (i32, f32) -> f32
        %result = "alex.sub"(%abc, %d) : (f32, f32) -> f32

        return %result : f32
    }


    func.func @TensorScalar() -> tensor<2x2xf32> {
        %a = "alex.constant"(){value = dense<1.0> : tensor<2x2xf32>}: () -> tensor<2x2xf32>
        %b = "alex.constant"(){value = 2.0 : f32}: () -> f32
        %c = "alex.constant"(){value = dense<3.0> : tensor<2x2xf32>}: () -> tensor<2x2xf32>
        %d = "alex.constant"(){value = 4.0 : f32}: () -> f32

        %bc = "alex.mul"(%b, %c): (f32, tensor<2x2xf32>) -> tensor<2x2xf32>
        %abc = "alex.add"(%a, %bc): (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %result = "alex.sub"(%abc, %d): (tensor<2x2xf32>, f32) -> tensor<2x2xf32>

        return %result : tensor<2x2xf32>
    }



}