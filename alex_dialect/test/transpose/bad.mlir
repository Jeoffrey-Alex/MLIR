func.func @bad(%x: tensor<2x3x4xf32>) -> tensor<3x2x4xf32> {
  %0 = "alex.transpose"(%x) {dim0 = 0 : i32, dim1 = 2 : i32}
      : (tensor<2x3x4xf32>) -> tensor<3x2x4xf32>
  return %0 : tensor<3x2x4xf32>
}
