module{
    func.func @add() -> tensor<2x2xf32> {
        %input1 = "arith.constant"() {
            value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
        } : () -> tensor<2x2xf32>

        %input2 = "arith.constant"() {
            value = dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
        } : () -> tensor<2x2xf32>

        %result = "alex.addLinalg"(%input1, %input2) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %result : tensor<2x2xf32>
    }
}

module{
    func.func @add() -> tensor<2x2xf32> {
        %input1 = "arith.constant"() {
            value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
        } : () -> tensor<2x2xf32>

        %input2 = "arith.constant"() {
            value = dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
        } : () -> tensor<2x2xf32>

        %result = "alex.addLinalg"(%input1, %input2) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %result : tensor<2x2xf32>
    }
}