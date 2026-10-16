module {
    func.func @relu_f32() -> tensor<2x3xf32>{
        %input = arith.constant dense<[[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]> : tensor<2x3xf32>
        %result = "alex.relu"(%input) : (tensor<2x3xf32>) -> tensor<2x3xf32>
        return %result : tensor<2x3xf32>
    }
}