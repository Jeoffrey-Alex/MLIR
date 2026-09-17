module {
    func.func @AddIntScalar() -> i32 attributes { llvm.emit_c_interface } {
        %a = "alex.constant"(){value = 5 : i32} : () -> i32
        %b = "alex.constant"(){value = 5 : i32} : () -> i32

        %result = "alex.add"(%a, %b) : (i32, i32) -> i32

        return %result : i32
    }

    func.func @AddFloatScalar() -> f32 attributes { llvm.emit_c_interface } {
        %a = "alex.constant"(){value = 5.0 : f32} : () -> f32
        %b = "alex.constant"(){value = 5.5 : f32} : () -> f32

        %result = "alex.add"(%a, %b) : (f32, f32) -> f32

        return %result : f32
    }

    func.func @AddIntFloatScalar() -> f32 attributes { llvm.emit_c_interface } {
        %a = "alex.constant"(){value = 5 : i32} : () -> i32
        %b = "alex.constant"(){value = 5.5 : f32} : () -> f32

        %result = "alex.add"(%a, %b) : (i32, f32) -> f32

        return %result : f32
    }

    func.func @AddIntTensors() -> tensor<2x2xi32> attributes { llvm.emit_c_interface } {
        %a = "alex.constant"(){value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
        %b = "alex.constant"(){value = dense<[[10, 20], [30, 40]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>

        %result = "alex.add"(%a, %b) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>

        return %result : tensor<2x2xi32>
    }

    func.func @AddFloatTensors() -> tensor<2x2xf32> attributes { llvm.emit_c_interface } {
        %a = "alex.constant"(){value = dense<[[1.1, 2.2], [3.3, 4.4]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
        %b = "alex.constant"(){value = dense<[[10.1, 20.2], [30.3, 40.4]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>

        %result = "alex.add"(%a, %b) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

        return %result : tensor<2x2xf32>
    }

    func.func @AddTensorScalar() -> tensor<2x2xf32> attributes { llvm.emit_c_interface } {
        %a = "alex.constant"(){value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>

        %scalar = "alex.constant"(){value = 5.0 : f32} : () -> f32

        %result = "alex.add"(%a, %scalar) : (tensor<2x2xf32>, f32) -> tensor<2x2xf32>

        return %result : tensor<2x2xf32>
    }
}