  // alex_add.mlir

  // RUN: alex-opt %s | FileCheck %s
  //CHECK-LABEL: func.func @add_with_optional
  //CHECK: "alex.add"
  //CHECK: testAttr
  //CHECK: "alex.add"(%arg0, %arg1) <{testAttr = [1.110000e+00 : f32, 2.220000e+00 : f32, 3.330000e+00 : f32]}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3x4xf32>

  func.func @add_with_optional(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3x4xf32> {
    %0 = "alex.add"(%arg0, %arg1) {
      testAttr = [1.11 : f32, 2.22 : f32, 3.33 : f32]
    } : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }

  //CHECK-LABEL: func.func @add_without_optional
  //CHECK: "alex.add"
  //CHECK: testAttr

  func.func @add_without_optional(%arg0: tensor<4x4xf32>) -> tensor<2x3x4xf32> {
    %0 = "alex.add"(%arg0) {
      testAttr = [1.000000e+00 : f32]
    } : (tensor<4x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }