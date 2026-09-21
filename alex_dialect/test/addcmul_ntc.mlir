// RUN: alex-opt %s -convert-alex-to-arith

module {
  func.func @AddcmulInvalid()->tensor<2x3xf32> attributes{
      llvm.emit_c_interface} {
    %input = "alex.constant"(){
      value = dense<[ [ 1.0, 2.0, 3.0 ], [ 4.0, 5.0, 6.0 ] ]> : tensor<2x3xf32>
    } : ()
            ->tensor<2x3xf32>

              %tensor1 = "alex.constant"(){
      value = dense<[ [ 2.0, 3.0 ], [ 4.0, 5.0 ] ]> : tensor<2x2xf32>
    } : ()
            ->tensor<2x2xf32>

                          %tensor2 = "alex.constant"(){
      value = dense<[ [ 5.0, 6.0, 7.0 ], [ 8.0, 9.0, 10.0 ] ]> : tensor<2x3xf32>
    } : ()
            ->tensor<2x3xf32>

                                      %
                                      result = "alex.addcmul"(
                                                   %input, %tensor1,
                                                   %tensor2){value = 2.0 : f32}
        : (tensor<2x3xf32>, tensor<2x2xf32>, tensor<2x3xf32>)
              ->tensor<2x3xf32>

          return %result : tensor<2x3xf32>
  }
}