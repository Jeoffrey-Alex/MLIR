module {
    func.func @AddTensor() -> tensor<2x2xf32> attributes { llvm.emit_c_interface } {
        %a = "alex.constant"() {value = dense<{{INPUT1}}> : tensor<2x2xf32>} : () -> tensor<2x2xf32>

        %b = "alex.constant"() {value = dense<{{INPUT2}}> : tensor<2x2xf32>} : () -> tensor<2x2xf32>

        %result = "alex.add"(%a, %b) :(tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

        return %result : tensor<2x2xf32>
    }
}