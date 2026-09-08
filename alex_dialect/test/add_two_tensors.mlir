module {
  %a = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]>
      : tensor<2x2xf32>

  %b = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]>
      : tensor<2x2xf32>

  // alpha omitted -> default value 1.0
  %result1 = "alex.add"(%a, %b)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

  // alpha explicitly specified
  %result2 = "alex.add"(%a, %b) {
      alpha = 2.0 : f32
  } : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
}