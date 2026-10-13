module {
  func.func @reduce_sum() -> i32 {
    %input = "arith.constant"() {
      value = dense<[10, 20, 30]> : tensor<3xi32>
    } : () -> tensor<3xi32>

    %result = "alex.reductionLinalg"(%input)
      : (tensor<3xi32>) -> i32

    return %result : i32
  }
}

