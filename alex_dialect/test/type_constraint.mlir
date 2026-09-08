// alex_add.mlir

func.func @add_with_optional(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3x4xf32> {
  %0 = "alex.add"(%arg0, %arg1) {
    testAttr = [1.000000e+00 : f32, 2.000000e+00 : f32, 3.000000e+00 : f32]
  } : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

func.func @add_without_optional(%arg0: tensor<4x4xf32>) -> tensor<2x3x4xf32> {
  %0 = "alex.add"(%arg0) {
    testAttr = [1.000000e+00 : f32]
  } : (tensor<4x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}