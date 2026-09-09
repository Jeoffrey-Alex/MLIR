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

    func.func@Tensor() ->   tensor<2x2xf32> {
        %a = "alex.constant"(){value = dense<1.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
        %b = "alex.constant"(){value = dense<2.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
        %c = "alex.constant"(){value = dense<3.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
        %d = "alex.constant"(){value = dense<4.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>

        %bc = "alex.mul"(%b, %c) : (tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
        %abc = "alex.add"(%a, %bc) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %result = "alex.sub"(%abc, %d) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

        return %result : tensor<2x2xf32>
    }

}